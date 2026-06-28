import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention and evidence experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--directory", type=Path, default=Path("data"))
    inspect = commands.add_parser("inspect-data")
    inspect.add_argument("--data", type=Path, default=Path("data/pubmedqa.json"))
    bench = commands.add_parser("benchmark")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--length", type=int, default=128)
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--output", type=Path, default=Path("artifacts/attention.json"))
    train = commands.add_parser("train-decoder")
    train.add_argument("--data", type=Path, default=Path("data/pubmedqa.json"))
    train.add_argument("--output", type=Path, default=Path("artifacts/decoder"))
    train.add_argument("--device", default="cpu")
    train.add_argument("--steps", type=int)
    train.add_argument("--accumulation", type=int, default=16)
    train.add_argument("--resume", type=Path)
    train.add_argument("--smoke", action="store_true")
    evidence = commands.add_parser("train-evidence")
    evidence.add_argument("--data", type=Path, default=Path("data/pubmedqa.json"))
    evidence.add_argument("--output", type=Path, default=Path("artifacts/evidence"))
    evidence.add_argument("--device", default="cpu")
    evidence.add_argument("--steps", type=int, default=200)
    evidence.add_argument("--accumulation", type=int, default=16)
    evidence.add_argument("--budget", type=int, default=256)
    evaluate = commands.add_parser("evaluate-evidence")
    evaluate.add_argument("--data", type=Path, default=Path("data/pubmedqa.json"))
    evaluate.add_argument("--checkpoint", type=Path, default=Path("artifacts/evidence/adapters.pt"))
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/budgets.json"))
    args = parser.parse_args()
    if args.command == "download":
        from pythonic.data import download_pubmedqa, load_papers, save_split, split_papers

        path = download_pubmedqa(args.directory)
        save_split(split_papers(load_papers(path)), args.directory / "split.json")
        print(path)
    elif args.command == "benchmark":
        from pythonic.experiments import attention_benchmark, save_json

        save_json(args.output, attention_benchmark(args.device, args.length, args.repeats))
    elif args.command == "inspect-data":
        from collections import Counter

        from pythonic.data import load_papers, split_papers

        split = split_papers(load_papers(args.data))
        for name in ("train", "validation", "test"):
            papers = getattr(split, name)
            print(name, len(papers), dict(Counter(paper.decision for paper in papers)))
    elif args.command == "train-decoder":
        import torch

        from pythonic.data import fixtures, load_papers, split_papers
        from pythonic.model import DecoderConfig
        from pythonic.training import TrainConfig, train_decoder

        torch.set_num_threads(4)
        if args.smoke:
            train_papers = validation = fixtures()
            model_config = DecoderConfig(width=32, layers=1, context=32)
        else:
            split = split_papers(load_papers(args.data))
            train_papers, validation = split.train, split.validation
            model_config = DecoderConfig()
        config = TrainConfig(
            steps=args.steps or (4 if args.smoke else 1000),
            accumulation=1 if args.smoke else args.accumulation,
            device=args.device,
        )
        _, report = train_decoder(
            train_papers, validation, model_config, config, args.output, args.resume
        )
        print("validation loss:", report["validation_loss"])
    elif args.command == "train-evidence":
        import torch

        from pythonic.biomedical import load_biomedical, train_evidence
        from pythonic.data import load_papers, split_papers
        from pythonic.training import TrainConfig

        torch.manual_seed(42)
        torch.set_num_threads(4)
        split = split_papers(load_papers(args.data))
        model, tokenizer = load_biomedical(args.device)
        config = TrainConfig(steps=args.steps, accumulation=args.accumulation, device=args.device)
        report = train_evidence(
            model, tokenizer, split.train, split.validation, config, args.output, args.budget
        )
        print(report["validation"])
    elif args.command == "evaluate-evidence":
        import hashlib

        import torch

        from pythonic.biomedical import load_evidence_checkpoint
        from pythonic.budgets import budget_experiment
        from pythonic.data import load_papers, split_papers
        from pythonic.experiments import save_json

        torch.set_num_threads(4)
        split = split_papers(load_papers(args.data))
        model, tokenizer = load_evidence_checkpoint(args.checkpoint, args.device)
        report = budget_experiment(model, tokenizer, split.test)
        report["adapter_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        save_json(args.output, report)
        print(args.output)


if __name__ == "__main__":
    main()
