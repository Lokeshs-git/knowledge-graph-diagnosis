import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from graph_diagnostic.features.extractor import FeatureExtractor
from graph_diagnostic.pipeline.generation import GraphGenerator
from graph_diagnostic.pipeline.retrieval import GraphRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def run_retrieval(qa_path: str, graph_path: str, degraded_dir: str | None = None, out: str = "experiments/runs/retrieval_cache.parquet") -> None:
    """
    Phase A: retrieval + feature extraction for all graph variants.

    Keyword extraction (LLM call) runs once per question and is reused
    across all 7 variants, eliminating 6/7 of the LLM calls in this phase.
    """
    qa_path = Path(qa_path)
    graph_base = Path(graph_path)
    out_file = Path(out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    base_name = graph_base.stem
    degraded_dir = Path(degraded_dir) if degraded_dir else graph_base.parent / "degraded"

    graphs_to_test = {
        "clean": graph_base,
        "light_mix": degraded_dir / f"{base_name}_light_mix.pkl",
        "moderate_mix": degraded_dir / f"{base_name}_moderate_mix.pkl",
        "heavy_mix": degraded_dir / f"{base_name}_heavy_mix.pkl",
        "ablation_schema": degraded_dir / f"{base_name}_ablation_schema.pkl",
        "ablation_resolution": degraded_dir / f"{base_name}_ablation_resolution.pkl",
        "ablation_fragmentation": degraded_dir / f"{base_name}_ablation_fragmentation.pkl",
    }

    logger.info(f"Loading QA pairs from {qa_path}...")
    qa_pairs = []
    with open(qa_path) as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                qa_pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed QA pair at line {line_idx + 1}: {e}")

    if not qa_pairs:
        logger.error("No valid QA pairs loaded. Aborting.")
        return

    # ── Step 1: Extract keywords ONCE per question (uses clean graph + LLM) ───
    logger.info("\n======================================")
    logger.info("Step 1: Keyword extraction (once per question, shared across variants)")
    logger.info("======================================")

    clean_emb = graph_base.with_name(f"{graph_base.stem}_embeddings.npy")
    with open(graph_base, "rb") as f:
        clean_graph = pickle.load(f)
    keyword_retriever = GraphRetriever(clean_graph, embeddings_path=clean_emb)

    keyword_cache: dict[int, tuple[list[str], str]] = {}
    for i, qa in enumerate(qa_pairs):
        logger.info(f"  [{i + 1}/{len(qa_pairs)}] {qa['question'][:80]}")
        keyword_cache[i] = keyword_retriever._extract_keywords(qa["question"])

    logger.info(f"Keyword extraction complete for {len(keyword_cache)} questions.")

    # ── Step 2: Retrieval + features per variant ──────────────────────────────
    rows = []

    for variant_name, g_path in graphs_to_test.items():
        if not g_path.exists():
            logger.warning(f"Variant {variant_name} not found at {g_path}, skipping.")
            continue

        variant_emb_path = g_path.with_name(f"{g_path.stem}_embeddings.npy")
        if not variant_emb_path.exists():
            logger.error(
                f"Embeddings missing for {variant_name} at {variant_emb_path}. "
                "Run precompute_embeddings.py first. Skipping."
            )
            continue

        logger.info("\n======================================")
        logger.info(f"Step 2: Retrieving — variant: {variant_name.upper()}")
        logger.info("======================================")

        with open(g_path, "rb") as f:
            graph = pickle.load(f)
        retriever = GraphRetriever(graph, embeddings_path=variant_emb_path)

        for i, qa in enumerate(qa_pairs):
            query = qa["question"]
            gold_ids = set(qa["gold_ids"])

            logger.info(f"  [{i + 1}/{len(qa_pairs)}] {query[:80]}")

            subgraph, seeds = retriever.retrieve_subgraph(
                query, k_seeds=3, hops=3, keywords_override=keyword_cache[i]
            )
            features = FeatureExtractor.extract_all(subgraph, seeds)
            context_str = GraphGenerator.format_subgraph_context(subgraph)
            retrieved_nodes = list(subgraph.nodes())

            logger.info(
                f"    nodes={subgraph.number_of_nodes()} "
                f"edges={subgraph.number_of_edges()} "
                f"seeds={len(seeds)}"
            )

            rows.append({
                "variant": variant_name,
                "query_idx": i,
                "query": query,
                "gold_ids": json.dumps(sorted(gold_ids)),
                "subgraph_context": context_str,
                "retrieved_nodes": json.dumps(retrieved_nodes),
                **features,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(out_file, index=False)
    logger.info(f"\nRetrieval phase complete! Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa",           default="data/finreflectkgqa/production_final.jsonl")
    parser.add_argument("--graph",        default="data/subsets/sp20_sample.pkl")
    parser.add_argument("--degraded-dir", default=None,
                        help="Path to degraded graph PKLs (default: <graph_parent>/degraded/)")
    parser.add_argument("--out",          default="experiments/runs/retrieval_cache.parquet",
                        help="Output path for retrieval cache parquet")
    args = parser.parse_args()

    run_retrieval(args.qa, args.graph, degraded_dir=args.degraded_dir, out=args.out)
