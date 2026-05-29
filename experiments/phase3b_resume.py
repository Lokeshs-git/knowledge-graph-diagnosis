"""Resume phase3b for specific variants whose generation failed mid-run.

Loads the existing results CSV, identifies rows with empty predictions for the
specified variants, re-runs generation for those rows from the retrieval cache,
and writes the merged result back to the same CSV.

Usage:
    GENERATION_PROVIDER=openrouter \\
    uv run python experiments/phase3b_resume.py \\
        --cache  experiments/runs/exp2_retrieval_cache.parquet \\
        --csv    experiments/runs/exp2_phase3_dataset.csv \\
        --variants ablation_schema ablation_resolution ablation_fragmentation
"""

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


def _generate_one(generator: GraphGenerator, idx: int, row: dict, total: int) -> dict:
    query = row["query"]
    context_str = row["subgraph_context"]
    gold_ids = set(json.loads(row["gold_ids"]))

    for attempt in range(_MAX_RETRIES):
        try:
            predicted_ids = set(generator.generate_from_context(query, context_str))
            metrics = compute_metrics(predicted_ids, gold_ids)
            logger.info(
                f"  [{idx + 1}/{total}] {row['variant']} Q{row['query_idx']} "
                f"F1={metrics['f1']:.2f} pred={sorted(predicted_ids)} gold={sorted(gold_ids)}"
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
                logger.warning(f"  [{idx + 1}/{total}] attempt {attempt + 1} failed: {e}. Retrying in {wait:.0f}s...")
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


def run_resume(cache_path: str, csv_path: str, variants: list[str], workers: int = _MAX_WORKERS) -> None:
    cache_path = Path(cache_path)
    csv_path   = Path(csv_path)

    logger.info(f"Loading retrieval cache from {cache_path}...")
    cache = pd.read_parquet(cache_path)

    logger.info(f"Loading existing results from {csv_path}...")
    existing = pd.read_csv(csv_path)

    # Rows to redo: target variants where predicted_ids is empty (generation failed)
    target_mask = (
        existing["variant"].isin(variants) &
        (existing["predicted_ids"].fillna("[]") == "[]")
    )
    failed_count = target_mask.sum()
    logger.info(
        f"Variants to resume: {variants}\n"
        f"  Rows with empty predictions in those variants: {failed_count} / {len(existing)}"
    )

    if failed_count == 0:
        logger.info("Nothing to resume — all rows already have predictions.")
        return

    # Pull subgraph_context from cache for the failed rows
    cache_key = cache.set_index(["variant", "query_idx"])
    failed_rows = existing[target_mask].copy()

    rows_to_run = []
    for _, row in failed_rows.iterrows():
        key = (row["variant"], int(row["query_idx"]))
        if key in cache_key.index:
            cache_row = cache_key.loc[key]
            row_dict = row.to_dict()
            row_dict["subgraph_context"] = cache_row["subgraph_context"]
            rows_to_run.append(row_dict)
        else:
            logger.warning(f"Cache miss for {key} — skipping.")

    total = len(rows_to_run)
    logger.info(f"Re-running generation for {total} rows with {workers} parallel workers...")

    generator = GraphGenerator()
    results = [None] * total

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_generate_one, generator, i, row, total): i
            for i, row in enumerate(rows_to_run)
        }
        for completed, future in enumerate(as_completed(future_to_idx), start=1):
            idx = future_to_idx[future]
            results[idx] = future.result()
            if completed % 50 == 0:
                logger.info(f"Progress: {completed}/{total} rows done")

    # Merge resumed rows back into existing DataFrame
    resumed_df = pd.DataFrame(results)
    resumed_df = resumed_df.set_index(["variant", "query_idx"])

    existing = existing.set_index(["variant", "query_idx"])
    existing.update(resumed_df[["predicted_ids", "f1_score", "precision", "recall"]])
    existing = existing.reset_index()

    existing.to_csv(csv_path, index=False)

    mean_f1 = existing["f1_score"].mean()
    by_variant = existing.groupby("variant")["f1_score"].mean().sort_index()
    logger.info(f"\nMerge complete. Updated {csv_path}")
    logger.info(f"Overall mean F1: {mean_f1:.3f}")
    for variant, f1 in by_variant.items():
        logger.info(f"  {variant:<28} F1={f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",    default="experiments/runs/exp2_retrieval_cache.parquet")
    parser.add_argument("--csv",      default="experiments/runs/exp2_phase3_dataset.csv")
    parser.add_argument("--variants", nargs="+",
                        default=["ablation_schema", "ablation_resolution", "ablation_fragmentation"])
    parser.add_argument("--workers",  type=int, default=_MAX_WORKERS)
    args = parser.parse_args()

    run_resume(args.cache, args.csv, args.variants, args.workers)
