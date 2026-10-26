"""
MoE Expert Paging System (mini-AGI style)

Manages expert weights stored on SSD with a resident pool in VRAM.
Uses safetensors memory-mapped loading for fast expert swapping.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from collections import OrderedDict
import safetensors.torch
import json
import hashlib


class ExpertIndex:
    """
    Index of all experts on disk.
    Maps (layer_idx, expert_id) -> file path, metadata.
    """
    
    def __init__(self, expert_dir: str, n_layers: int, n_experts: int):
        self.expert_dir = expert_dir
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.index_path = os.path.join(expert_dir, "expert_index.json")
        self._index: Dict[str, Dict] = {}
        self._load_or_create()
    
    def _load_or_create(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)
        else:
            # Create default index
            os.makedirs(self.expert_dir, exist_ok=True)
            for layer_idx in range(self.n_layers):
                for expert_id in range(self.n_experts):
                    key = f"layer{layer_idx}_expert{expert_id}"
                    filename = f"{key}.safetensors"
                    self._index[key] = {
                        "path": filename,
                        "layer": layer_idx,
                        "expert": expert_id,
                        "size_bytes": 0,
                        "checksum": "",
                    }
            self.save()
    
    def save(self):
        with open(self.index_path, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def get_path(self, layer_idx: int, expert_id: int) -> str:
        key = f"layer{layer_idx}_expert{expert_id}"
        return os.path.join(self.expert_dir, self._index[key]["path"])
    
    def update_metadata(self, layer_idx: int, expert_id: int, size_bytes: int, checksum: str):
        key = f"layer{layer_idx}_expert{expert_id}"
        self._index[key]["size_bytes"] = size_bytes
        self._index[key]["checksum"] = checksum
        self.save()


class MoEPager:
    """
    Manages expert residency in VRAM.
    
    - Expert weights stored on SSD as safetensors files
    - Resident pool: LRU cache of expert weights in VRAM
    - Async loading with CUDA streams (when available)
    - Memory-mapped reads for fast access
    """
    
    def __init__(
        self,
        expert_dir: str,
        resident_slots: int = 32,
        n_layers: int = 12,
        n_experts: int = 64,
        d_model: int = 768,
        expert_hidden: int = 3072,
        device: Optional[torch.device] = None,
    ):
        self.expert_dir = expert_dir
        self.resident_slots = resident_slots
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.d_model = d_model
        self.expert_hidden = expert_hidden
        self.device = device or torch.device('cpu')
        
        # Index of all experts on disk
        self.index = ExpertIndex(expert_dir, n_layers, n_experts)
        
        # Resident pool: OrderedDict (LRU) of (layer, expert) -> dict of tensors
        self.resident: OrderedDict[Tuple[int, int], Dict[str, torch.Tensor]] = OrderedDict()
        
        # Track which experts are currently loaded for each layer
        self.layer_resident: List[Set[int]] = [set() for _ in range(n_layers)]
        
        # Loading stream for async (CPU only for now)
        self._load_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    def _expert_key(self, layer_idx: int, expert_id: int) -> Tuple[int, int]:
        return (layer_idx, expert_id)
    
    def _load_expert_from_disk(self, layer_idx: int, expert_id: int) -> Dict[str, torch.Tensor]:
        """Load expert weights from safetensors file."""
        path = self.index.get_path(layer_idx, expert_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert not found: {path}")
        
        # Memory-mapped load
        tensors = safetensors.torch.load_file(path, device=str(self.device))
        return tensors
    
    def _evict_lru(self):
        """Evict least recently used expert from resident pool."""
        if not self.resident:
            return
        
        (layer_idx, expert_id), tensors = self.resident.popitem(last=False)
        self.layer_resident[layer_idx].discard(expert_id)
        
        # Free GPU memory
        for tensor in tensors.values():
            del tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def ensure_resident(self, layer_idx: int, expert_ids: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Ensure experts are resident in VRAM.
        Returns dict of expert_id -> tensors for the requested experts.
        """
        result = {}
        
        for expert_id in expert_ids:
            key = self._expert_key(layer_idx, expert_id)
            
            if key in self.resident:
                # Already resident - move to end (MRU)
                tensors = self.resident.pop(key)
                self.resident[key] = tensors
                result[expert_id] = tensors
            else:
                # Need to load
                if len(self.resident) >= self.resident_slots:
                    self._evict_lru()
                
                tensors = self._load_expert_from_disk(layer_idx, expert_id)
                self.resident[key] = tensors
                self.layer_resident[layer_idx].add(expert_id)
                result[expert_id] = tensors
        
        return result
    
    def get_resident_experts(self, layer_idx: int) -> Set[int]:
        """Get set of currently resident expert IDs for a layer."""
        return self.layer_resident[layer_idx].copy()
    
    def save_expert(self, layer_idx: int, expert_id: int, tensors: Dict[str, torch.Tensor]):
        """Save expert weights to disk (checkpointing)."""
        path = self.index.get_path(layer_idx, expert_id)
        
        # Compute checksum (convert BF16 to float32 first)
        all_bytes = b""
        for name in sorted(tensors.keys()):
            t = tensors[name]
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            all_bytes += t.cpu().numpy().tobytes()
        checksum = hashlib.sha256(all_bytes).hexdigest()[:16]
        
        # Save (safetensors supports BF16)
        safetensors.torch.save_file(tensors, path)
        size_bytes = os.path.getsize(path)
        
        # Update index
        self.index.update_metadata(layer_idx, expert_id, size_bytes, checksum)
    
    def save_all_resident(self):
        """Save all resident experts to disk (periodic checkpoint)."""
        for (layer_idx, expert_id), tensors in self.resident.items():
            self.save_expert(layer_idx, expert_id, tensors)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage stats in MB."""
        resident_count = len(self.resident)
        # Each expert: gate + up + down = 3 * (d_model * expert_hidden) * 2 bytes (BF16)
        params_per_expert = 3 * self.d_model * self.expert_hidden
        bytes_per_expert = params_per_expert * 2  # BF16
        
        return {
            "resident_experts": resident_count,
            "resident_mb": resident_count * bytes_per_expert / 1e6,
            "total_experts": self.n_layers * self.n_experts,
            "total_mb": self.n_layers * self.n_experts * bytes_per_expert / 1e6,
        }
    
    def __repr__(self):
        return f"MoEPager(resident={len(self.resident)}/{self.resident_slots}, layers={self.n_layers}, experts={self.n_experts})"


def create_expert_tensors(d_model: int, expert_hidden: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Create randomly initialized expert tensors (gate, up, down for SwiGLU)."""
    return {
        "gate": torch.randn(expert_hidden, d_model, device=device, dtype=dtype) * 0.02,
        "up": torch.randn(expert_hidden, d_model, device=device, dtype=dtype) * 0.02,
        "down": torch.randn(d_model, expert_hidden, device=device, dtype=dtype) * 0.02,
    }


def swiglu_forward(x: torch.Tensor, expert_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    SwiGLU forward pass with expert tensors.
    
    Args:
        x: [*, d_model]
        expert_tensors: dict with 'gate', 'up', 'down' weights
    Returns:
        out: [*, d_model]
    """
    gate = F.linear(x, expert_tensors["gate"])  # [*, expert_hidden]
    up = F.linear(x, expert_tensors["up"])       # [*, expert_hidden]
    hidden = F.silu(gate) * up                   # [*, expert_hidden]
    down = F.linear(hidden, expert_tensors["down"])  # [*, d_model]
    return down


if __name__ == "__main__":
    # Quick test
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pager = MoEPager(tmpdir, resident_slots=4, n_layers=2, n_experts=8, d_model=64, expert_hidden=256)
        
        # Create and save some experts
        for layer in range(2):
            for expert in range(8):
                tensors = create_expert_tensors(64, 256, torch.device('cpu'), torch.bfloat16)
                pager.save_expert(layer, expert, tensors)
        
        # Load experts
        loaded = pager.ensure_resident(0, [0, 1, 2])
        print(f"Loaded experts: {list(loaded.keys())}")
        print(f"Resident: {pager.get_resident_experts(0)}")
        
        # Test forward
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        out = swiglu_forward(x, loaded[0])
        print(f"SwiGLU out: {out.shape}")
        
        # Test eviction
        loaded = pager.ensure_resident(0, [3, 4, 5, 6, 7])  # exceeds 4 slots
        print(f"After eviction, resident: {pager.get_resident_experts(0)}")
        
        print("MoEPager test passed!")