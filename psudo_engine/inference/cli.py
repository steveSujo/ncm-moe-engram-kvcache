#!/usr/bin/env python3
"""
CLI Demo Script for psudo-Engine.

Provides command-line interface for model inference, generation,
and interactive chat mode.
"""

import argparse
import json
import sys

import torch
from omegaconf import OmegaConf

from psudo_engine.inference.generator import GenerationConfig, PsudoGenerator
from psudo_engine.inference.tokenizer import PsudoTokenizer
from psudo_engine.model.psudo_transformer import PsudoTransformer, create_model


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments to parser."""
    parser.add_argument('--config', type=str, default='psudo_engine/config/model.yaml',
                        help='Path to model config')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--weight-store', type=str, default='weights',
                        help='Path to weight store directory')


def get_device(device_arg: str) -> torch.device:
    """Get torch device from argument."""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def load_model_and_tokenizer(
    config_path: str,
    checkpoint_path: str,
    device: torch.device
) -> tuple[PsudoTransformer, PsudoTokenizer]:
    """Load model and tokenizer from checkpoint."""
    config = OmegaConf.load(config_path)

    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device)
    model.prepare_for_inference()

    tokenizer = PsudoTokenizer(vocab_size=config.model.vocab_size)

    return model, tokenizer


def cmd_generate(args: argparse.Namespace):
    """Generate text from prompt."""
    device = get_device(args.device)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.config, args.checkpoint, device)

    # Create generator
    generator = PsudoGenerator(model=model, tokenizer=tokenizer, device=device)

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_tokens=[tokenizer.eos_token_id],
        repetition_penalty=args.repetition_penalty,
        stream=args.stream,
    )

    # Generate

    generator.generate(args.prompt, gen_config)


def cmd_chat(args: argparse.Namespace):
    """Interactive chat mode."""
    device = get_device(args.device)

    model, tokenizer = load_model_and_tokenizer(args.config, args.checkpoint, device)
    generator = PsudoGenerator(model=model, tokenizer=tokenizer, device=device)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_tokens=[tokenizer.eos_token_id],
        repetition_penalty=args.repetition_penalty,
    )


    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ('exit', 'quit'):
            break

        if user_input.lower() == 'clear':
            messages = []
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = generator.chat(messages, gen_config)
        messages.append({"role": "assistant", "content": response})


def cmd_benchmark(args: argparse.Namespace):
    """Run generation benchmark."""
    device = get_device(args.device)

    model, tokenizer = load_model_and_tokenizer(args.config, args.checkpoint, device)
    generator = PsudoGenerator(model=model, tokenizer=tokenizer, device=device)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stream=False,
    )

    prompts = [
        "fn hello() -> string {",
        "fn add(a: int, b: int) -> int {",
        "fn factorial(n: int) -> int {",
        "fn sum_range(start: int, end: int) -> int {",
    ]


    total_tokens = 0
    total_time = 0.0

    for _i, prompt in enumerate(prompts):
        generator.generate(prompt, gen_config)
        stats = generator.get_stats()
        total_tokens += stats['total_tokens']
        total_time += stats['total_time']



def cmd_tokenize(args: argparse.Namespace):
    """Tokenize input text."""
    config = OmegaConf.load(args.config)
    tokenizer = PsudoTokenizer(vocab_size=config.model.vocab_size)

    if args.input_file:
        with open(args.input_file) as f:
            text = f.read()
    else:
        text = args.text

    if args.encode:
        ids = tokenizer.encode(text)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(ids, f)
    else:
        if args.input_file:
            with open(args.input_file) as f:
                ids = json.load(f)
        else:
            ids = list(map(int, args.text.split()))
        decoded = tokenizer.decode(ids)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(decoded)


def cmd_info(args: argparse.Namespace):
    """Show model info."""
    config = OmegaConf.load(args.config)


    # Count parameters
    model = create_model(config)
    sum(p.numel() for p in model.parameters())
    sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description='psudo-Engine CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text
  python -m psudo_engine.inference.cli generate --prompt "fn hello() {" --checkpoint model.pt

  # Chat mode
  python -m psudo_engine.inference.cli chat --checkpoint model.pt

  # Benchmark
  python -m psudo_engine.inference.cli benchmark --checkpoint model.pt

  # Tokenize
  python -m psudo_engine.inference.cli tokenize --text "fn test() {}" --encode

  # Model info
  python -m psudo_engine.inference.cli info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text from prompt')
    add_common_args(gen_parser)
    gen_parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    gen_parser.add_argument('--max-tokens', type=int, default=100, help='Max new tokens')
    gen_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    gen_parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty')
    gen_parser.add_argument('--stream', action='store_true', help='Stream output')
    gen_parser.set_defaults(func=cmd_generate)

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    add_common_args(chat_parser)
    chat_parser.add_argument('--max-tokens', type=int, default=100, help='Max new tokens')
    chat_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    chat_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    chat_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    chat_parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty')
    chat_parser.set_defaults(func=cmd_chat)

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run generation benchmark')
    add_common_args(bench_parser)
    bench_parser.add_argument('--max-tokens', type=int, default=50, help='Max new tokens per prompt')
    bench_parser.set_defaults(func=cmd_benchmark)

    # Tokenize command
    tok_parser = subparsers.add_parser('tokenize', help='Encode/decode text')
    add_common_args(tok_parser)
    tok_parser.add_argument('--text', type=str, help='Input text or token IDs')
    tok_parser.add_argument('--input-file', type=str, help='Input file')
    tok_parser.add_argument('--output', type=str, help='Output file')
    tok_parser.add_argument('--encode', action='store_true', help='Encode text to tokens')
    tok_parser.add_argument('--decode', action='store_true', help='Decode tokens to text')
    tok_parser.set_defaults(func=cmd_tokenize)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show model info')
    add_common_args(info_parser)
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Handle device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        args.func(args)
    except Exception:
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
