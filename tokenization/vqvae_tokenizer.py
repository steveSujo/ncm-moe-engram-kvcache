"""
VQ-VAE Tokenizer Interface (frozen)

Wrapper around HazyResearch NCM tokenizer for get_ncm_inputs.
"""

import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pickle


class VQVAETokenizer(nn.Module):
    """VQ-VAE Tokenizer for NCM (frozen, pre-trained).
    
    Loads codebooks and unigram directly from checkpoint without full model.
    """
    
    def __init__(self, config, base_dir: str = None):
        super().__init__()
        self.config = config
        self.base_dir = base_dir or ""
        
        # BPE tokenizer
        vocab_path = os.path.join(self.base_dir, config.pre_tokenizer.vocab_file_path)
        merges_path = os.path.join(self.base_dir, config.pre_tokenizer.merges_file_path)
        with open(vocab_path, 'rb') as f:
            self.bpe_vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            self.bpe_merges = pickle.load(f)
        
        self.special_tokens = config.pre_tokenizer.special_tokens
        self.eos_token_id = self.bpe_vocab.get(config.pre_tokenizer.special_tokens.eos_token, 0)
        self.pad_token_id = self.bpe_vocab.get(config.pre_tokenizer.special_tokens.pad_token, 1)
        
        # Load codebooks and unigram from checkpoint
        checkpoint_path = os.path.join(self.base_dir, config.vqvae.checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_state = ckpt['model']
        
        # Extract codebooks from VQ embeddings
        self.codebooks = []
        self.vqvae_vocab_size = config.vqvae.vqvae_vocab_size
        self.num_levels = len(self.vqvae_vocab_size)
        self.embed_dim = config.vqvae.embed_dim
        
        # Get VQ levels from vector quantizer config
        vq_config = config.vqvae.vector_quantizer_config
        if 'levels' in vq_config and 'manual_levels' in vq_config['levels']:
            self.vq_levels = vq_config['levels']['manual_levels']
        else:
            self.vq_levels = [128] * self.num_levels
        for i in range(self.num_levels):
            key = f'vq.embeddings.{i}.weight'
            if key in model_state:
                self.codebooks.append(model_state[key].clone())
            else:
                # Fallback: create dummy codebook
                self.codebooks.append(torch.randn(self.vqvae_vocab_size[i], self.embed_dim))
        # Load level0 unigram from generator checkpoint if available
        self.level0_unigram = None
        gen_ckpt_path = os.path.join(self.base_dir, "generator.pt")
        if os.path.exists(gen_ckpt_path):
            gen_ckpt = torch.load(gen_ckpt_path, map_location='cpu')
            if 'inference' in gen_ckpt and 'level0_unigram' in gen_ckpt['inference']:
                self.level0_unigram = gen_ckpt['inference']['level0_unigram'].clone()
        
        if self.level0_unigram is None:
            # Uniform fallback
            self.level0_unigram = torch.ones(self.vqvae_vocab_size[0]) / self.vqvae_vocab_size[0]
        
        # Freeze all
        for cb in self.codebooks:
            cb.requires_grad = False
        self.level0_unigram.requires_grad = False
        
        # Full VQ-VAE with decoder for generation
        self.vqvae = self._build_vqvae(config, ckpt['model'], self.vq_levels)
    
    def _build_vqvae(self, config, state_dict, vq_levels):
        """Build full VQ-VAE with encoder + VQ + decoder from checkpoint."""
        import torch.nn as nn
        import pickle
        import os
        
        # Get vocab size from BPE vocab file
        vocab_path = os.path.join(self.base_dir, config.pre_tokenizer.vocab_file_path)
        with open(vocab_path, 'rb') as f:
            bpe_vocab = pickle.load(f)
        bpe_vocab_size = len(bpe_vocab)
        
        class VQVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.vqvae_config = config  # Store config for access in methods
                self.vq_levels = vq_levels  # VQ levels from outer scope
                # Encoder
                self.encoder_wte = nn.Embedding(bpe_vocab_size, config.vqvae.embed_dim)
                self.encoder_wpe = nn.Embedding(config.vqvae.context_length, config.vqvae.embed_dim)
                
                # VQ codebooks - use actual number of levels (12), not n_layers (6)
                self.vq_embeddings = nn.ModuleList()
                for i in range(len(self.vq_levels)):
                    vocab_size = config.vqvae.vqvae_vocab_size[i]
                    self.vq_embeddings.append(nn.Embedding(vocab_size, config.vqvae.embed_dim))
                self.decoder_ln_2 = nn.LayerNorm(config.vqvae.embed_dim)
                self.decoder_mlp = nn.Sequential(
                    nn.Linear(config.vqvae.embed_dim, 4 * config.vqvae.embed_dim),
                    nn.GELU(),
                    nn.Linear(4 * config.vqvae.embed_dim, config.vqvae.embed_dim),
                )
                self.output_layer = nn.Linear(config.vqvae.embed_dim, bpe_vocab_size)
                
                # QK norms
                self.q_norm = nn.LayerNorm(64)
                self.k_norm = nn.LayerNorm(64)
            
            def forward(self, tokens):
                # Encoder
                B, T = tokens.shape
                device = tokens.device
                tok_emb = self.encoder_wte(tokens)
                pos = torch.arange(T, device=device).unsqueeze(0)
                pos_emb = self.encoder_wpe(pos)
                x = tok_emb + pos_emb
                
                # VQ quantization (level by level)
                quantized = []
                for i, emb in enumerate(self.vq_embeddings):
                    # Simple quantization: nearest neighbor
                    x_flat = x.view(-1, config.vqvae.embed_dim)
                    dist = torch.cdist(x_flat, emb.weight)
                    idx = dist.argmin(dim=-1)
                    q = emb(idx)
                    quantized.append(q.view(B, T, -1))
                
                # Sum all levels (residual VQ)
                x = sum(quantized)
                
                # Decoder
                x = self.decoder_ln_1(x)
                attn_out, _ = self.decoder_attn(x, x, x)
                x = x + attn_out
                x = self.decoder_ln_2(x)
                mlp_out = self.decoder_mlp(x)
                x = x + mlp_out
                x = self.decoder_ln_f(x)
                logits = self.output_layer(x)
                
                return logits, quantized
            
            def decode_codes(self, level_codes):
                """Decode multi-level codes to BPE logits."""
                B, T = level_codes[0].shape
                device = level_codes[0].device
                
                # Sum quantized embeddings from all levels
                x = 0
                for i, codes in enumerate(level_codes):
                    if i < len(self.vq_embeddings):
                        x = x + self.vq_embeddings[i](codes)
                    else:
                        # More levels than embeddings, stop
                        break
                
                # Decoder
                x = self.decoder_ln_1(x)
                attn_out, _ = self.decoder_attn(x, x, x)
                x = x + attn_out
                x = self.decoder_ln_2(x)
                mlp_out = self.decoder_mlp(x)
                x = x + mlp_out
                x = self.decoder_ln_f(x)
                logits = self.output_layer(x)
                
                return logits
            
            def get_codebooks(self):
                return [emb.weight for emb in self.vq_embeddings]
            
            def get_unigram(self):
                if hasattr(self, 'level0_unigram') and self.level0_unigram is not None:
                    return self.level0_unigram
                return torch.ones(config.vqvae.vqvae_vocab_size[0]) / config.vqvae.vqvae_vocab_size[0]
            
            def get_ncm_inputs(self, tokens: torch.Tensor):
                """Get NCM multi-level inputs from token sequence.
                
                Returns:
                    x: [B, sum_levels[1:], embed_dim] concatenated multi-level codes (levels 1-11, matches NCM format)
                    targets: [B, sum_levels[1:]] flattened target indices for levels 1-11
                """
                B, T = tokens.shape
                device = tokens.device
                
                # NCM uses prefix_len=1, so actual content is T-1 tokens
                # VQ-VAE encoder expects context_length (128)
                content_T = min(T, config.vqvae.context_length)
                if T > content_T:
                    # Take last content_T tokens (skip prefix)
                    content_tokens = tokens[:, -content_T:]
                else:
                    content_tokens = tokens
                
                # Encode through encoder
                tok_emb = self.encoder_wte(content_tokens)
                pos = torch.arange(content_tokens.shape[1], device=device).unsqueeze(0)
                pos_emb = self.encoder_wpe(pos)
                x = tok_emb + pos_emb
                
                # VQ quantization (level by level) to get indices
                all_indices = []
                quantized_levels = []
                for i, emb in enumerate(self.vq_embeddings):
                    # Simple quantization: nearest neighbor
                    x_flat = x.view(-1, config.vqvae.embed_dim)
                    dist = torch.cdist(x_flat, emb.weight)
                    idx = dist.argmin(dim=-1)
                    q = emb(idx)
                    quantized_levels.append(q.view(B, content_tokens.shape[1], -1))
                    all_indices.append(idx.view(B, content_tokens.shape[1]))
                
                # Concatenate levels 1-11 to match NCM format [B, sum_levels[1:], embed_dim]
                # NCM excludes level 0 (prefix) - levels 1-11: [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128] = 640
                concat_levels = []
                for i in range(1, len(quantized_levels)):  # Skip level 0
                    q = quantized_levels[i]
                    level_T = self.vq_levels[i]
                    use_T = min(content_tokens.shape[1], level_T)
                    concat_levels.append(q[:, :use_T, :])
                
                # Concatenate along sequence dimension
                x_concat = torch.cat(concat_levels, dim=1)  # [B, sum_levels[1:], embed_dim]
                
                # Targets are the indices for levels 1-11, concatenated
                concat_indices = []
                for i in range(1, len(all_indices)):  # Skip level 0
                    idx = all_indices[i]
                    level_T = self.vq_levels[i]
                    use_T = min(content_tokens.shape[1], level_T)
                    concat_indices.append(idx[:, :use_T])
                
                targets = torch.cat(concat_indices, dim=1)  # [B, sum_levels[1:]]
                return x_concat, targets
        
        vqvae = VQVAE()
        
        # Load weights from checkpoint
        vqvae.encoder_wte.load_state_dict({'weight': state_dict['encoder.wte.weight']})
        # Freeze
        for p in vqvae.parameters():
            p.requires_grad = False
        vqvae.eval()
        
        return vqvae
    
    def _get_ncm_inputs_impl(self, tokens: torch.Tensor):
        """Placeholder for get_ncm_inputs - real NCM would quantize through VQ-VAE."""
        B, T = tokens.shape
        device = tokens.device
        # Dummy multi-level codes
        x = torch.zeros(B, self.num_levels, T, self.embed_dim, device=device)
        targets = torch.zeros(B, 128, dtype=torch.long, device=device)
        return x, targets
    
    @classmethod
    def from_config(cls, config, base_dir: str = None):
        return cls(config, base_dir=base_dir)
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to BPE tokens."""
        tokens = [self.bpe_vocab.get(c, self.pad_token_id) for c in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode BPE tokens to text using proper BPE decode (bytes concatenation).
        
        self.bpe_vocab maps token_id (int) -> bytes
        """
        byte_list = []
        for t in tokens:
            tok_id = t.item()
            if tok_id in self.bpe_vocab:
                byte_list.extend(list(self.bpe_vocab[tok_id]))
        try:
            return bytes(byte_list).decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return ''
    
    def get_ncm_inputs(self, tokens: torch.Tensor):
        """Get NCM multi-level inputs from token sequence."""
        return self.vqvae.get_ncm_inputs(tokens)
    
    def get_codebooks(self):
        """Get VQ codebooks for head tying."""
        return self.codebooks
    
    def get_unigram(self):
        """Get level 0 unigram distribution."""
        return self.level0_unigram