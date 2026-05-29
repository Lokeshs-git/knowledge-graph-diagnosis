import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from graph_diagnostic.evaluation.metrics import compute_metrics
from graph_diagnostic.features.extractor import FeatureExtractor
from graph_diagnostic.pipeline.generation import GraphGenerator
from graph_diagnostic.pipeline.retrieval import GraphRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(qa_path: str, graph_path: str):
    """
    Runs the QA workload against all 4 graph variants.
    """
    qa_path = Path(qa_path)
    out_dir = Path("experiments/runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define the 4 graph variants dynamically based on base graph path
    graph_base = Path(graph_path)
    base_name = graph_base.stem
    degraded_dir = graph_base.parent / "degraded"

    graphs_to_test = {
        "clean": graph_base,
        "light_mix": degraded_dir / f"{base_name}_light_mix.pkl",
        "moderate_mix": degraded_dir / f"{base_name}_moderate_mix.pkl",
        "heavy_mix": degraded_dir / f"{base_name}_heavy_mix.pkl",
        "ablation_schema": degraded_dir / f"{base_name}_ablation_schema.pkl",
        "ablation_resolution": degraded_dir / f"{base_name}_ablation_resolution.pkl",
        "ablation_fragmentation": degraded_dir / f"{base_name}_ablation_fragmentation.pkl"
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

    generator = GraphGenerator()
    dataset_rows = []

    for variant_name, g_path in graphs_to_test.items():
        if not g_path.exists():
            logger.warning(f"Variant {variant_name} not found at {g_path}, skipping.")
            continue

        logger.info("\n======================================")
        logger.info(f"Running pipeline on variant: {variant_name.upper()}")
        logger.info("======================================")

        with open(g_path, "rb") as f:
            graph = pickle.load(f)

        # Each variant has its own embeddings (node sets differ for duplicate-entity variants)
        variant_emb_path = g_path.with_name(g_path.stem + "_embeddings.npy")
        if not variant_emb_path.exists():
            logger.error(f"Embeddings missing for {variant_name} at {variant_emb_path}. Run precompute_embeddings.py first.")
            continue
        retriever = GraphRetriever(graph, embeddings_path=variant_emb_path)

        for i, qa in enumerate(qa_pairs):
            query = qa["question"]
            gold_ids = set(qa["gold_ids"])

            logger.info(f"Q {i+1}/{len(qa_pairs)}: {query}")

            # 1. Retrieve
            logger.info("  [Trace] Starting retrieval...")
            subgraph, seeds = retriever.retrieve_subgraph(query, k_seeds=3, hops=3)
            logger.info(f"  [Trace] Retrieval done. Subgraph size: {subgraph.number_of_nodes()} nodes.")

            # 2. Extract ML Features
            logger.info("  [Trace] Starting feature extraction...")
            features = FeatureExtractor.extract_all(subgraph, seeds)
            logger.info("  [Trace] Feature extraction done.")

            # 3. Generate Answer
            logger.info("  [Trace] Calling LLM for generation...")
            predicted_ids = set(generator.generate_answer(query, subgraph))

            # 4. Evaluate
            metrics = compute_metrics(predicted_ids, gold_ids)
            f1 = metrics["f1"]

            retrieved_nodes = list(subgraph.nodes())
            logger.info(f"  Predicted IDs : {predicted_ids}")
            logger.info(f"  Gold IDs      : {gold_ids}")
            logger.info(f"  F1 Score      : {f1:.2f}")
            logger.info(f"  Nodes retrieved: {len(retrieved_nodes)} | Edges: {subgraph.number_of_edges()}")
            logger.info(f"  Features      : nodes={features['node_count']}, entropy={features['entity_diversity']:.2f}, fill={features['property_fill_rate']:.2f}")

            dataset_rows.append({
                "variant": variant_name,
                "query_idx": i,
                "query": query,
                "gold_ids": json.dumps(sorted(gold_ids)),
                "predicted_ids": json.dumps(sorted(predicted_ids)),
                "retrieved_nodes": json.dumps(retrieved_nodes),
                "f1_score": f1,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                **features,
            })

    df = pd.DataFrame(dataset_rows)
    out_file = out_dir / "phase3_dataset.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"\nDataset generation complete! Saved {len(df)} rows to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="data/finreflectkgqa/production_final.jsonl")
    parser.add_argument("--graph", default="data/subsets/sp20_sample.pkl")
    args = parser.parse_args()

    run_experiment(args.qa, args.graph)
