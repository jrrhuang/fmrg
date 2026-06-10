"""Print per-tag 4-seed mean GenEval Overall scores."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def overall_score(jsonl: Path) -> float:
    by_tag = defaultdict(list)
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            by_tag[r["tag"]].append(int(r["correct"]))
    if not by_tag:
        return float("nan")
    return sum(sum(v) / len(v) for v in by_tag.values()) / len(by_tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_root", type=Path)
    args = p.parse_args()

    seed_re = re.compile(r"_seed(\d+)$")
    per_tag = defaultdict(dict)
    for results in args.results_root.rglob("results.jsonl"):
        m = seed_re.search(results.parent.name)
        if not m:
            continue
        tag = seed_re.sub("", results.parent.name)
        per_tag[tag][int(m.group(1))] = overall_score(results)

    if not per_tag:
        print(f"No results.jsonl found under {args.results_root}", file=sys.stderr)
        return

    print(f"{'tag':30s} {'n':>3}  mean   per-seed")
    for tag, scores in sorted(per_tag.items()):
        mean = sum(scores.values()) / len(scores)
        per_seed = " ".join(f"{s}:{v:.4f}" for s, v in sorted(scores.items()))
        print(f"{tag:30s} {len(scores):>3}  {mean:.4f}  {per_seed}")


if __name__ == "__main__":
    main()
