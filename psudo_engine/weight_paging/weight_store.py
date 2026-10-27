"""
Weight Store - Disk-backed layer chunk storage for psudo-Engine.

Manages serialization/deserialization of model layer weights to/from disk
in a format compatible with Mini-AGI paging and boot-time unpacking.
"""

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LayerChunkMetadata:
    """Metadata for a single layer chunk on disk."""
    layer_idx: int
    chunk_name: str
    shape: list[int]
    dtype: str
    numel: int
    size_bytes: int
    checksum: str
    offset: int
    quantized: bool
    quantization_params: dict[str, Any] | None = None


@dataclass
class ModelIndex:
    """Index of all layer chunks for a model."""
    model_name: str
    version: str
    total_layers: int
    total_size_bytes: int
    chunks: list[LayerChunkMetadata]
    created_at: str


def compute_checksum(tensor: torch.Tensor) -> str:
    """Compute SHA256 checksum of tensor data."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def tensor_to_dtype_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to string."""
    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype))


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return mapping.get(dtype_str, torch.float32)


class WeightStore:
    """
    Disk-backed weight store for model layers.

    Stores each layer as a separate chunk file (.bin) with metadata index.
    Supports both full-precision and quantized (ternary/2-bit) formats.
    """

    def __init__(self, store_dir: str, model_name: str = "psudo_engine"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.index_path = self.store_dir / f"{model_name}_index.json"
        self.chunks_dir = self.store_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)

        self._index: ModelIndex | None = None
        self._load_index()

    def _load_index(self):
        """Load existing index or create new."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            chunks = [LayerChunkMetadata(**c) for c in data['chunks']]
            self._index = ModelIndex(
                model_name=data['model_name'],
                version=data['version'],
                total_layers=data['total_layers'],
                total_size_bytes=data['total_size_bytes'],
                chunks=chunks,
                created_at=data['created_at']
            )
        else:
            self._index = ModelIndex(
                model_name=self.model_name,
                version="1.0.0",
                total_layers=0,
                total_size_bytes=0,
                chunks=[],
                created_at=""
            )

    def _save_index(self):
        """Save index to disk."""
        import datetime
        if self._index.created_at == "":
            self._index.created_at = datetime.datetime.utcnow().isoformat() + "Z"

        data = {
            'model_name': self._index.model_name,
            'version': self._index.version,
            'total_layers': self._index.total_layers,
            'total_size_bytes': self._index.total_size_bytes,
            'chunks': [asdict(c) for c in self._index.chunks],
            'created_at': self._index.created_at
        }
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_layer(
        self,
        layer_idx: int,
        tensors: dict[str, torch.Tensor],
        quantized: bool = False,
        quantization_params: dict[str, Any] | None = None
    ) -> list[LayerChunkMetadata]:
        """
        Save a layer's tensors to disk as separate chunks.

        Args:
            layer_idx: Layer index
            tensors: Dict of tensor_name -> tensor
            quantized: Whether tensors are quantized
            quantization_params: Quantization parameters if quantized

        Returns:
            List of chunk metadata for saved tensors
        """
        saved_chunks = []

        for name, tensor in tensors.items():
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # Ensure contiguous
            tensor = tensor.contiguous()

            # Compute metadata
            shape = list(tensor.shape)
            dtype_str = tensor_to_dtype_str(tensor.dtype)
            numel = tensor.numel()
            size_bytes = numel * tensor.element_size()
            checksum = compute_checksum(tensor)

            # Generate chunk filename
            chunk_name = f"layer_{layer_idx}_{name}"
            chunk_path = self.chunks_dir / f"{chunk_name}.bin"

            # Determine offset (append to file or new file)
            # For simplicity, each tensor gets its own file
            offset = 0

            # Write tensor data
            with open(chunk_path, 'wb') as f:
                # Write header: shape length, shape, dtype
                f.write(struct.pack('<I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('<I', dim))
                dtype_bytes = dtype_str.encode('utf-8')
                f.write(struct.pack('<I', len(dtype_bytes)))
                f.write(dtype_bytes)
                # Write tensor data
                f.write(tensor.numpy().tobytes())

            # Create metadata
            meta = LayerChunkMetadata(
                layer_idx=layer_idx,
                chunk_name=chunk_name,
                shape=shape,
                dtype=dtype_str,
                numel=numel,
                size_bytes=size_bytes,
                checksum=checksum,
                offset=offset,
                quantized=quantized,
                quantization_params=quantization_params
            )

            saved_chunks.append(meta)
            self._index.chunks.append(meta)

        # Update index
        self._index.total_layers = max(self._index.total_layers, layer_idx + 1)
        self._index.total_size_bytes = sum(c.size_bytes for c in self._index.chunks)
        self._save_index()

        return saved_chunks

    def load_layer(self, layer_idx: int, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """
        Load all tensors for a specific layer.

        Args:
            layer_idx: Layer index to load
            device: Target device (default: cpu)

        Returns:
            Dict of tensor_name -> tensor
        """
        if device is None:
            device = torch.device('cpu')

        layer_tensors = {}

        for chunk_meta in self._index.chunks:
            if chunk_meta.layer_idx == layer_idx:
                chunk_path = self.chunks_dir / f"{chunk_meta.chunk_name}.bin"

                if not chunk_path.exists():
                    raise FileNotFoundError(f"Chunk not found: {chunk_path}")

                with open(chunk_path, 'rb') as f:
                    # Read header
                    shape_len = struct.unpack('<I', f.read(4))[0]
                    shape = []
                    for _ in range(shape_len):
                        shape.append(struct.unpack('<I', f.read(4))[0])

                    dtype_len = struct.unpack('<I', f.read(4))[0]
                    dtype_str = f.read(dtype_len).decode('utf-8')
                    dtype = dtype_str_to_torch(dtype_str)

                    # Read tensor data
                    numel = chunk_meta.numel
                    data = f.read(numel * torch.tensor([], dtype=dtype).element_size())

                    tensor = torch.frombuffer(data, dtype=dtype).reshape(shape)
                    tensor = tensor.to(device)

                    # Verify checksum
                    if compute_checksum(tensor.cpu()) != chunk_meta.checksum:
                        raise ValueError(f"Checksum mismatch for {chunk_meta.chunk_name}")

                    # Extract tensor name from chunk_name
                    # format: layer_{idx}_{name}
                    parts = chunk_meta.chunk_name.split('_', 2)
                    tensor_name = parts[2] if len(parts) >= 3 else chunk_meta.chunk_name

                    layer_tensors[tensor_name] = tensor

        return layer_tensors

    def load_tensor(self, layer_idx: int, tensor_name: str, device: torch.device | None = None) -> torch.Tensor:
        """Load a single tensor by layer and name."""
        chunk_name = f"layer_{layer_idx}_{tensor_name}"

        for chunk_meta in self._index.chunks:
            if chunk_meta.chunk_name == chunk_name:
                chunk_path = self.chunks_dir / f"{chunk_name}.bin"

                with open(chunk_path, 'rb') as f:
                    shape_len = struct.unpack('<I', f.read(4))[0]
                    shape = [struct.unpack('<I', f.read(4))[0] for _ in range(shape_len)]
                    dtype_len = struct.unpack('<I', f.read(4))[0]
                    dtype_str = f.read(dtype_len).decode('utf-8')
                    dtype = dtype_str_to_torch(dtype_str)

                    numel = chunk_meta.numel
                    data = f.read(numel * torch.tensor([], dtype=dtype).element_size())

                    tensor = torch.frombuffer(data, dtype=dtype).reshape(shape)

                    if device is not None:
                        tensor = tensor.to(device)

                    if compute_checksum(tensor.cpu()) != chunk_meta.checksum:
                        raise ValueError(f"Checksum mismatch for {chunk_name}")

                    return tensor

        raise KeyError(f"Tensor not found: layer_{layer_idx}_{tensor_name}")

    def get_layer_chunks(self, layer_idx: int) -> list[LayerChunkMetadata]:
        """Get metadata for all chunks in a layer."""
        return [c for c in self._index.chunks if c.layer_idx == layer_idx]

    def get_all_layers(self) -> list[int]:
        """Get list of all layer indices stored."""
        layers = {c.layer_idx for c in self._index.chunks}
        return sorted(layers)

    def layer_exists(self, layer_idx: int) -> bool:
        """Check if a layer exists in the store."""
        return any(c.layer_idx == layer_idx for c in self._index.chunks)

    def evict_layer(self, layer_idx: int) -> int:
        """
        Remove a layer's chunks from disk (eviction).

        Returns:
            Number of bytes freed
        """
        freed = 0
        chunks_to_remove = []

        for chunk_meta in self._index.chunks:
            if chunk_meta.layer_idx == layer_idx:
                chunk_path = self.chunks_dir / f"{chunk_meta.chunk_name}.bin"
                if chunk_path.exists():
                    freed += chunk_path.stat().st_size
                    chunk_path.unlink()
                chunks_to_remove.append(chunk_meta)

        for chunk in chunks_to_remove:
            self._index.chunks.remove(chunk)

        self._index.total_size_bytes = sum(c.size_bytes for c in self._index.chunks)
        if self._index.chunks:
            self._index.total_layers = max(c.layer_idx for c in self._index.chunks) + 1
        else:
            self._index.total_layers = 0

        self._save_index()
        return freed

    def get_memory_usage(self) -> dict[str, int]:
        """Get storage usage statistics."""
        total_chunks = len(self._index.chunks)
        total_size = self._index.total_size_bytes
        layers = self.get_all_layers()

        return {
            'total_chunks': total_chunks,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'num_layers': len(layers),
            'layers': layers
        }

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify all chunks match their checksums."""
        errors = []

        for chunk_meta in self._index.chunks:
            chunk_path = self.chunks_dir / f"{chunk_meta.chunk_name}.bin"
            if not chunk_path.exists():
                errors.append(f"Missing chunk file: {chunk_meta.chunk_name}")
                continue

            try:
                tensor = self.load_tensor(chunk_meta.layer_idx, chunk_meta.chunk_name.split('_', 2)[2])
                if compute_checksum(tensor.cpu()) != chunk_meta.checksum:
                    errors.append(f"Checksum mismatch: {chunk_meta.chunk_name}")
            except Exception as e:
                errors.append(f"Error loading {chunk_meta.chunk_name}: {e}")

        return len(errors) == 0, errors


def save_model_to_store(
    model: torch.nn.Module,
    store: WeightStore,
    quantize: bool = False
) -> ModelIndex:
    """
    Save entire model to weight store, layer by layer.

    Args:
        model: The model to save
        store: WeightStore instance
        quantize: Whether to quantize weights before saving

    Returns:
        Updated ModelIndex
    """
    # Group parameters by layer
    layer_params: dict[int, dict[str, torch.Tensor]] = {}

    for name, param in model.named_parameters():
        # Parse layer index from parameter name
        # Expected format: blocks.{layer_idx}.{rest}
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == 'blocks':
            try:
                layer_idx = int(parts[1])
                tensor_name = '.'.join(parts[2:])
            except ValueError:
                layer_idx = -1  # Embedding or other
                tensor_name = name
        else:
            layer_idx = -1
            tensor_name = name

        if layer_idx not in layer_params:
            layer_params[layer_idx] = {}
        layer_params[layer_idx][tensor_name] = param.data.clone()

    # Save each layer
    for layer_idx, tensors in layer_params.items():
        q_params = None
        if quantize:
            # TODO: Apply quantization before saving
            pass
        store.save_layer(layer_idx, tensors, quantized=quantize, quantization_params=q_params)

    return store._index


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = WeightStore(tmpdir, "test_model")

        # Create test tensors
        tensors = {
            'attn.q_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
            'attn.k_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
            'attn.v_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
            'attn.o_proj.weight': torch.randn(512, 512, dtype=torch.bfloat16),
            'mlp.w1.weight': torch.randn(2048, 512, dtype=torch.bfloat16),
            'mlp.w2.weight': torch.randn(512, 2048, dtype=torch.bfloat16),
            'mlp.w3.weight': torch.randn(2048, 512, dtype=torch.bfloat16),
        }

        # Save layer 0
        store.save_layer(0, tensors)

        # Load back
        loaded = store.load_layer(0)

        # Verify
        for name, orig in tensors.items():
            loaded_tensor = loaded[name]
            assert torch.allclose(orig, loaded_tensor), f"Mismatch in {name}"

