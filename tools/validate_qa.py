"""Validate the QA set against the updated retriever.

Samples --n questions, runs the full pipeline, and exits 0 if mean F1 >= --threshold.
Used by remote_pipeline.sh to decide whether to use the existing QA file or regenerate.
"""

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

from graph_diagnostic.evaluation.metrics import compute_metrics
from graph_diagnostic.pipeline.generation import GraphGenerator
from graph_diagnostic.pipeline.retrieval import GraphRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate(qa_path: str, graph_path: str, embeddings_path: str, n: int, threshold: float) -> bool:
    qa_path = Path(qa_path)
    graph_path = Path(graph_path)
    embeddings_path = Path(embeddings_path)

    logger.info(f"Loading QA pairs from {qa_path}...")
    all_qa: list[dict] = []
    with open(qa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_qa.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not all_qa:
        logger.error("No valid QA pairs found.")
        return False

    sample = random.sample(all_qa, min(n, len(all_qa)))
    logger.info(f"Sampled {len(sample)} questions for validation.")

    logger.info(f"Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    retriever = GraphRetriever(graph, embeddings_path=embeddings_path)
    generator = GraphGenerator()

    scores: list[float] = []
    for i, qa in enumerate(sample):
        query = qa["question"]
        gold_ids = set(qa["gold_ids"])

        subgraph, _seeds = retriever.retrieve_subgraph(query, k_seeds=3, hops=3)
        predicted_ids = set(generator.generate_answer(query, subgraph))
        metrics = compute_metrics(predicted_ids, gold_ids)
        f1 = metrics["f1"]
        scores.append(f1)
        logger.info(f"  [{i + 1}/{len(sample)}] F1={f1:.2f} | pred={predicted_ids} | gold={gold_ids}")

    mean_f1 = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"Validation mean F1: {mean_f1:.3f} (threshold: {threshold})")

    return mean_f1 >= threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate QA set against updated retriever")
    parser.add_argument("--qa", default="data/finreflectkgqa/production_final.jsonl")
    parser.add_argument("--graph", default="data/subsets/sp20_sample.pkl")
    parser.add_argument("--embeddings", default="data/subsets/sp20_sample_embeddings.npy")
    parser.add_argument("--n", type=int, default=25, help="Number of questions to sample")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum acceptable mean F1")
    args = parser.parse_args()

    passed = validate(args.qa, args.graph, args.embeddings, args.n, args.threshold)
    sys.exit(0 if passed else 1)
