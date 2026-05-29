"""Generate additional QA pairs and combine with an existing JSONL file.

Usage:
    uv run python tools/generate_additional_qa.py \\
        --graph data/subsets/sp20_sample.pkl \\
        --n 750 \\
        --existing data/finreflectkgqa/production_final.jsonl \\
        --additional data/finreflectkgqa/production_additional_750.jsonl \\
        --combined data/finreflectkgqa/production_exp2_1000.jsonl
"""

import argparse
import logging
from pathlib import Path

from graph_diagnostic.data.qa_generator import QAGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def combine(existing: Path, additional: Path, combined: Path) -> int:
    lines = []
    for src in (existing, additional):
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    with open(combined, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return len(lines)


def run(graph: str, n: int, existing: str, additional: str, combined: str) -> None:
    existing_path   = Path(existing)
    additional_path = Path(additional)
    combined_path   = Path(combined)

    if additional_path.exists():
        with open(additional_path) as _f:
            existing_count = sum(1 for _ in _f)
        logger.info(f"[SKIP] {additional_path} already exists ({existing_count} lines).")
    else:
        logger.info(f"Generating {n} additional QA pairs → {additional_path}")
        generator = QAGenerator(graph)
        generator.generate_batch(n=n, output_file=additional_path)
        logger.info(f"[OK] {n} pairs saved to {additional_path}")

    if combined_path.exists():
        logger.info(f"[SKIP] Combined file {combined_path} already exists.")
    else:
        total = combine(existing_path, additional_path, combined_path)
        logger.info(f"[OK] Combined {total} QA pairs → {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",      default="data/subsets/sp20_sample.pkl")
    parser.add_argument("--n",          type=int, default=750)
    parser.add_argument("--existing",   default="data/finreflectkgqa/production_final.jsonl")
    parser.add_argument("--additional", default="data/finreflectkgqa/production_additional_750.jsonl")
    parser.add_argument("--combined",   default="data/finreflectkgqa/production_exp2_1000.jsonl")
    args = parser.parse_args()

    run(args.graph, args.n, args.existing, args.additional, args.combined)
