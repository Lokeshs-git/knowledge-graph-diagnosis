import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from graph_diagnostic.evaluation.metrics import compute_metrics
from graph_diagnostic.pipeline.generation import GraphGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

_MAX_WORKERS = 10
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 5.0


def _generate_one(
    generator: GraphGenerator,
    idx: int,
    row: dict,
    total: int,
) -> dict:
    query = row["query"]
    context_str = row["subgraph_context"]
    gold_ids = set(json.loads(row["gold_ids"]))

    for attempt in range(_MAX_RETRIES):
        try:
            predicted_ids = set(generator.generate_from_context(query, context_str))
            metrics = compute_metrics(predicted_ids, gold_ids)
            logger.info(
                f"  [{idx + 1}/{total}] {row['variant']} Q{row['query_idx']} "
                f"F1={metrics['f1']:.2f} "
                f"pred={sorted(predicted_ids)} "
                f"gold={sorted(gold_ids)}"
            )
            return {
                **row,
                "predicted_ids": json.dumps(sorted(predicted_ids)),
                "f1_score": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY * (attempt + 1)
                logger.warning(
                    f"  [{idx + 1}/{total}] attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"  [{idx + 1}/{total}] all {_MAX_RETRIES} attempts failed: {e}")
                return {
                    **row,
                    "predicted_ids": json.dumps([]),
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }


def run_generation(cache_path: str, workers: int = _MAX_WORKERS, out: str | None = None) -> None:
    """
    Phase B: parallel LLM generation + evaluation.

    Reads retrieval_cache.parquet, fires all generation calls in parallel using
    a thread pool, then writes phase3_dataset.csv.
    """
    cache_path = Path(cache_path)
    out_file = Path(out) if out else cache_path.parent / "phase3_dataset.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading retrieval cache from {cache_path}...")
    df = pd.read_parquet(cache_path)
    total = len(df)
    logger.info(f"Loaded {total} rows. Generating with {workers} parallel workers...")

    generator = GraphGenerator()
    rows = df.to_dict(orient="records")
    results: list[dict | None] = [None] * total

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_generate_one, generator, i, row, total): i
            for i, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(future_to_idx), start=1):
            idx = future_to_idx[future]
            results[idx] = future.result()
            if completed % 50 == 0:
                logger.info(f"Progress: {completed}/{total} rows done")

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_file, index=False)

    mean_f1 = result_df["f1_score"].mean()
    by_variant = result_df.groupby("variant")["f1_score"].mean().sort_index()
    logger.info(f"\nGeneration phase complete! Saved {len(result_df)} rows to {out_file}")
    logger.info(f"Overall mean F1: {mean_f1:.3f}")
    for variant, f1 in by_variant.items():
        logger.info(f"  {variant:<28} F1={f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",   default="experiments/runs/retrieval_cache.parquet")
    parser.add_argument("--workers", type=int, default=_MAX_WORKERS)
    parser.add_argument("--out",     default=None,
                        help="Output CSV path (default: <cache_dir>/phase3_dataset.csv)")
    args = parser.parse_args()

    run_generation(args.cache, args.workers, out=args.out)
