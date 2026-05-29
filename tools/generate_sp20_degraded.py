"""Generate all six degraded variants of the SP20 graph.

Skips any variant whose .pkl already exists — safe to re-run.
Saves each corrupted graph and its provenance JSON to --out-dir.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

from graph_diagnostic.corruption.simulate import GraphCorruptor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Rates derived from PROJECT.md degradation spec (lower/mid/upper bounds)
VARIANTS: dict[str, dict[str, float]] = {
    "light_mix": {
        "missing_properties": 0.15,
        "duplicate_entities": 0.02,
        "flipped_relations": 0.005,
        "orphan_subgraphs": 0.005,
    },
    "moderate_mix": {
        "missing_properties": 0.275,
        "duplicate_entities": 0.05,
        "flipped_relations": 0.0175,
        "orphan_subgraphs": 0.005,
    },
    "heavy_mix": {
        "missing_properties": 0.40,
        "duplicate_entities": 0.08,
        "flipped_relations": 0.03,
        "orphan_subgraphs": 0.01,
    },
    "ablation_schema": {
        "missing_properties": 0.40,
        "duplicate_entities": 0.0,
        "flipped_relations": 0.0,
        "orphan_subgraphs": 0.0,
    },
    "ablation_resolution": {
        "missing_properties": 0.0,
        "duplicate_entities": 0.08,
        "flipped_relations": 0.0,
        "orphan_subgraphs": 0.0,
    },
    "ablation_fragmentation": {
        "missing_properties": 0.0,
        "duplicate_entities": 0.0,
        "flipped_relations": 0.0,
        "orphan_subgraphs": 0.01,
    },
}


def generate(graph_path: str, out_dir: str, scale: float = 1.0) -> None:
    graph_path = Path(graph_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = graph_path.stem  # e.g. "sp20_sample"

    logger.info(f"Loading base graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        base_graph = pickle.load(f)
    logger.info(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    corruptor = GraphCorruptor(random_seed=42)

    if scale != 1.0:
        logger.info(f"Applying scale factor {scale:.3f} to all degradation rates.")

    for variant_name, rates in VARIANTS.items():
        out_pkl = out_dir / f"{base_name}_{variant_name}.pkl"
        out_prov = out_dir / f"{base_name}_{variant_name}_provenance.json"

        if out_pkl.exists():
            logger.info(f"[SKIP] {variant_name} already exists at {out_pkl}")
            continue

        scaled_rates = {k: round(min(v * scale, 1.0), 6) for k, v in rates.items()}
        logger.info(f"[GEN] Generating {variant_name} with rates: {scaled_rates}")
        rates = scaled_rates
        corrupted_graph, provenance = corruptor.apply_degradation(base_graph, rates)

        with open(out_pkl, "wb") as f:
            pickle.dump(corrupted_graph, f)

        with open(out_prov, "w") as f:
            json.dump(provenance, f, indent=2)

        logger.info(
            f"[OK] {variant_name}: {corrupted_graph.number_of_nodes()} nodes, "
            f"{corrupted_graph.number_of_edges()} edges → {out_pkl}"
        )

    logger.info("Done. All variants generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate degraded SP20 graph variants")
    parser.add_argument("--graph",   default="data/subsets/sp20_sample.pkl")
    parser.add_argument("--out-dir", default="data/subsets/degraded")
    parser.add_argument("--scale",   type=float, default=1.0,
                        help="Multiply all degradation rates by this factor (e.g. 1.33 for 33%% more extreme)")
    args = parser.parse_args()
    generate(args.graph, args.out_dir, scale=args.scale)
