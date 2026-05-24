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


if __name__ == "__main__":
    main()
