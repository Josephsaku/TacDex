"""Run with python -m tacdex, or the installed tacdex command."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Single-ID3 TactSpace research pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synthetic", help="Generate clearly marked software fixtures")
    s.add_argument("--out", required=True)
    s.add_argument("--frames", type=int, default=40)
    s.add_argument("--seed", type=int, default=7)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", required=True)
    for command in ("pretrain", "probe"):
        t = sub.add_parser(command)
        t.add_argument("--manifest", required=True)
        t.add_argument("--config", required=True)
        t.add_argument("--out", required=True)
        t.add_argument("--device", default="auto")
        if command == "probe":
            t.add_argument("--pretrained", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--manifest", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", default="test_id")
    e.add_argument("--device", default="auto")
    e.add_argument("--out", required=True)
    i = sub.add_parser("inspect-replay")
    i.add_argument("path")
    args = p.parse_args()
    if args.command == "synthetic":
        from .synthetic import generate
        print(generate(args.out, args.frames, args.seed))
    elif args.command == "validate":
        from .data import read_manifest
        m = read_manifest(args.manifest)
        print(json.dumps({"valid": True, "episodes": len(m["episodes"])}, indent=2))
    elif args.command == "pretrain":
        from .training import pretrain
        pretrain(args.manifest, args.config, args.out, args.device)
    elif args.command == "probe":
        from .training import train_probe
        train_probe(args.manifest, args.pretrained, args.config, args.out, args.device)
    elif args.command == "evaluate":
        from .training import evaluate
        result = evaluate(args.manifest, args.checkpoint, args.split, args.device)
        path = Path(args.out)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        from .adapters import inspect_replay
        print(json.dumps(inspect_replay(args.path), indent=2))
