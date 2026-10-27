#!/usr/bin/env python3
"""
Benchmarks for psudo-Engine.

Measures:
- VRAM usage (peak, average)
- Latency (tokens/sec, ms/token)
- Quality (perplexity, compile success rate)
"""

import json
from omegaconf import OmegaConf
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib

from psudo_engine.context.context_builder import ContextBuilder, ContextConfig
from psudo_engine.context.retriever import HybridRetriever
from psudo_engine.context.syntax_hub import SyntaxHub
from psudo_engine.inference.generator import GenerationConfig, PsudoGenerator
from psudo_engine.inference.tokenizer import PsudoTokenizer
from psudo_engine.model.psudo_transformer import PsudoTransformer, create_model


@dataclass
class VRAMStats:
    """VRAM usage statistics."""
    peak_mb: float
    avg_mb: float
    allocated_mb: float
    reserved_mb: float

@dataclass
class LatencyStats:
    """Latency statistics."""
    tokens_per_sec: float
    ms_per_token: float
    prefill_ms: float
    decode_ms_per_token: float
    total_time_sec: float

@dataclass
class QualityStats:
    """Quality metrics."""
    perplexity: float | None
    compile_success_rate: float | None
    syntax_valid_rate: float | None
    compile_errors: list[str]

@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    model_config: dict[str, Any]
    vram: VRAMStats
    latency: LatencyStats
    quality: QualityStats
    timestamp: str


def get_vram_stats() -> VRAMStats:
    """Get current VRAM statistics."""
    if not torch.cuda.is_available():
        return VRAMStats(0, 0, 0, 0)

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return VRAMStats(
        peak_mb=peak,
        avg_mb=allocated,
        allocated_mb=allocated,
        reserved_mb=reserved
    )


def reset_vram_stats():
    """Reset VRAM statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def measure_prefill_latency(model: PsudoTransformer, input_ids: torch.Tensor,
                            use_cache: bool = False) -> float:
    """Measure prefill latency (first forward pass)."""
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        _ = model.forward(input_ids, use_cache=use_cache, return_dict=True)

    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000  # ms


def measure_generation_latency(generator: PsudoGenerator, prompt: str,
                               max_new_tokens: int = 50,
                               config: GenerationConfig | None = None) -> LatencyStats:
    """Measure generation latency."""
    tokenizer = generator.tokenizer
    input_ids = tokenizer.encode(prompt)
    input_ids = [tokenizer.bos_token_id] + input_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=generator.device)

    config = config or GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        use_cache=True
    )

    generator.model.reset_kv_cache()

    # Prefill timing
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()

    outputs = generator.model.forward(input_tensor, use_cache=config.use_cache, return_dict=True)

    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    prefill_ms = (prefill_end - prefill_start) * 1000

    # Generation timing
    current_ids = input_tensor.clone()
    decode_times = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            torch.cuda.synchronize()
            decode_start = time.perf_counter()

            outputs = generator.model.forward(
                current_ids[:, -1:],
                use_cache=config.use_cache,
                return_dict=True
            )

            torch.cuda.synchronize()
            decode_end = time.perf_counter()
            decode_times.append((decode_end - decode_start) * 1000)

            logits = outputs['logits']
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            if next_token.item() == generator.tokenizer.eos_token_id:
                break

            if current_ids.shape[1] >= generator.model.max_seq_len:
                break

            # Ensure next_token has correct shape [batch, 1] for concatenation
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if current_ids.shape[1] >= generator.model.max_seq_len:
                break

    total_time = sum(decode_times) + prefill_ms
    tokens_generated = len(decode_times)

    return LatencyStats(
        tokens_per_sec=1000 * tokens_generated / sum(decode_times) if decode_times else 0,
        ms_per_token=statistics.mean(decode_times) if decode_times else 0,
        prefill_ms=prefill_ms,
        decode_ms_per_token=statistics.mean(decode_times) if decode_times else 0,
        total_time_sec=total_time / 1000
    )


def evaluate_quality(generator: PsudoGenerator, prompts: list[str],
                     transpiler=None) -> QualityStats:
    """Evaluate generation quality (compile success, syntax validity)."""
    if transpiler is None:
        # Without transpiler, just check syntax validity
        results = {"valid": 0, "total": 0, "errors": []}

        for prompt in prompts:
            result = generator.generate(prompt, GenerationConfig(max_new_tokens=50))
            # Basic syntax check
            results["total"] += 1
            if "fn" in result and "{" in result and "}" in result:
                results["valid"] += 1
            else:
                results["errors"].append(f"Invalid syntax: {result[:100]}")

        return QualityStats(
            perplexity=None,
            compile_success_rate=None,
            syntax_valid_rate=results["valid"] / results["total"] if results["total"] > 0 else 0,
            compile_errors=results["errors"]
        )

    # With transpiler - would attempt actual compilation
    # Placeholder for future implementation
    return QualityStats(
        perplexity=None,
        compile_success_rate=None,
        syntax_valid_rate=0.0,
        compile_errors=["Transpiler not implemented"]
    )


def run_model_benchmark(config: dict[str, Any], prompts: list[str],
                        max_new_tokens: int = 50) -> BenchmarkResult:
    """Run complete benchmark suite."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_model(config)
    model.to(device)
    model.prepare_for_inference()

    tokenizer = PsudoTokenizer(vocab_size=config.model.vocab_size)
    generator = PsudoGenerator(model=model, tokenizer=tokenizer)

    reset_vram_stats()

    # VRAM baseline
    vram_baseline = get_vram_stats()

    # Test prompt
    test_prompt = prompts[0] if prompts else "fn test() {"
    input_ids = generator.tokenizer.encode(test_prompt)
    input_ids = [generator.tokenizer.bos_token_id] + input_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Measure prefill
    measure_prefill_latency(generator.model, input_tensor, use_cache=True)

    # VRAM after prefill
    vram_after_prefill = get_vram_stats()

    # Generation latency
    gen_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        use_cache=True
    )
    latency_stats = measure_generation_latency(generator, "fn test() {",
                                               max_new_tokens=50, config=gen_config)

    # VRAM after generation
    vram_after_gen = get_vram_stats()

    # Quality (basic syntax check)
    quality = evaluate_quality(generator, ["fn test() {", "fn add(a, b) { return a + b }"])

    peak_mb = max(vram_baseline.peak_mb, vram_after_prefill.peak_mb, vram_after_gen.peak_mb)

    return BenchmarkResult(
        model_config=config,
        vram=VRAMStats(
            peak_mb=peak_mb,
            avg_mb=vram_after_gen.avg_mb,
            allocated_mb=vram_after_gen.allocated_mb,
            reserved_mb=vram_after_gen.reserved_mb
        ),
        latency=latency_stats,
        quality=quality,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


def run_context_benchmark() -> dict[str, Any]:
    """Benchmark context retrieval system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub = SyntaxHub(tmpdir)
        hub.initialize(create_defaults=True)

        retriever = HybridRetriever()
        retriever.add_document("func1", "fn read_file(path: string) -> string { read_file(path) }",
                              {"language": "psudo", "type": "builtin"})
        retriever.add_document("func2", "fn write_file(path: string, content: string) -> () { write_file(path, content) }",
                              {"language": "psudo", "type": "builtin"})
        retriever.build()

        builder = ContextBuilder(
            config=ContextConfig(max_tokens=1500),
            syntax_hub=hub,
            retriever=retriever
        )

        # Benchmark retrieval
        torch.cuda.synchronize()
        start = time.perf_counter()

        context = builder.build_context(
            task="Read a file and print its contents",
            language='psudo'
        )

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        return {
            "retrieval_time_ms": elapsed,
            "context_length": len(context),
            "context_tokens_est": len(context) // 4
        }


def save_benchmark_result(result: BenchmarkResult, output_path: str):
    """Save benchmark result to JSON."""
    from omegaconf import OmegaConf
    
    output = {
        "model_config": OmegaConf.to_container(result.model_config, resolve=True),
        "vram": asdict(result.vram),
        "latency": asdict(result.latency),
        "quality": asdict(result.quality),
        "timestamp": result.timestamp
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    """Run all benchmarks."""

    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        pass

    # Test configurations
    configs = {
        "small": {
            "model": {
                "vocab_size": 1000,
                "d_model": 256,
                "n_layers": 2,
                "n_heads": 4,
                "head_dim": 32,
                "d_ff": 1024,
                "n_experts": 4,
                "top_k": 2,
                "resident_slots": 2,
                "engram_layers": [1],
                "engram_slots": 1024,
                "engram_heads": 2,
                "engram_orders": [2, 3],
                "max_window": 128,
                "sink_tokens": 1,
                "weight_bits": 1.58,
                "tie_embeddings": True,
                "max_seq_len": 256,
                "dropout": 0.1,
            }
        },
        "medium": {
            "model": {
                "vocab_size": 8192,
                "d_model": 512,
                "n_layers": 12,
                "n_heads": 8,
                "head_dim": 64,
                "d_ff": 2048,
                "n_experts": 64,
                "top_k": 2,
                "resident_slots": 32,
                "engram_layers": [4, 12, 20],
                "engram_slots": 8192,
                "engram_heads": 4,
                "engram_orders": [2, 3],
                "max_window": 256,
                "sink_tokens": 1,
                "weight_bits": 1.58,
                "tie_embeddings": True,
                "max_seq_len": 2048,
                "dropout": 0.1,
            }
        }
    }

    prompts = [
        "fn hello() -> string {",
        "fn add(a: int, b: int) -> int { return a + b }",
        "fn factorial(n: int) -> int { if n <= 1 { return 1 } return n * factorial(n - 1) }",
        "fn read_config(path: string) -> string { return read_file(path) }",
    ]

    results = []

    for name, config_dict in configs.items():

        try:
            # Convert dict to OmegaConf DictConfig
            config = OmegaConf.create(config_dict)
            result = run_model_benchmark(config, prompts)
            results.append(result)

            print(f"  VRAM Peak: {result.vram.peak_mb:.1f} MB")
            print(f"  Latency: {result.latency.tokens_per_sec:.1f} tokens/sec")
            print(f"  Prefill: {result.latency.prefill_ms:.1f} ms")
            print(f"  Decode: {result.latency.decode_ms_per_token:.2f} ms/token")
            print(f"  Syntax Valid: {result.quality.syntax_valid_rate:.1%}")

            # Save individual result
            save_benchmark_result(result, f"benchmarks/result_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Context benchmark

    with contextlib.suppress(Exception):
        run_context_benchmark()

    # Summary
    for _r in results:
        pass


if __name__ == "__main__":
    main()
