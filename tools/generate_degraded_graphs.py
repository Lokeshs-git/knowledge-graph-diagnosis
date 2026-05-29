import logging
import pickle
import json
from pathlib import Path
from graph_diagnostic.corruption.simulate import GraphCorruptor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    graph_path = Path("data/subsets/aapl_sample.pkl")
    out_dir = Path("data/subsets/degraded")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(graph_path, "rb") as f:
        clean_graph = pickle.load(f)
        
    corruptor = GraphCorruptor(random_seed=42)
    
    # Define degradation variants (Mixed and Isolated)
    variants = {
        "light_mix": {
            "missing_properties": 0.05,
            "duplicate_entities": 0.01,
            "flipped_relations": 0.005,
            "orphan_subgraphs": 0.01
        },
        "moderate_mix": {
            "missing_properties": 0.15,
            "duplicate_entities": 0.05,
            "flipped_relations": 0.02,
            "orphan_subgraphs": 0.05
        },
        "heavy_mix": {
            "missing_properties": 0.30,
            "duplicate_entities": 0.10,
            "flipped_relations": 0.05,
            "orphan_subgraphs": 0.10
        },
        # Isolated Ablation Variants for clear SHAP mapping
        "ablation_schema": {
            "missing_properties": 0.40,
            "duplicate_entities": 0.0,
            "flipped_relations": 0.0,
            "orphan_subgraphs": 0.0
        },
        "ablation_resolution": {
            "missing_properties": 0.0,
            "duplicate_entities": 0.15,
            "flipped_relations": 0.0,
            "orphan_subgraphs": 0.0
        },
        "ablation_fragmentation": {
            "missing_properties": 0.0,
            "duplicate_entities": 0.0,
            "flipped_relations": 0.0,
            "orphan_subgraphs": 0.20
        }
    }
    
    for name, rates in variants.items():
        logging.info(f"Generating {name} variant...")
        corrupted, provenance = corruptor.apply_degradation(clean_graph, rates)
        
        # Save graph
        with open(out_dir / f"aapl_sample_{name}.pkl", "wb") as f:
            pickle.dump(corrupted, f)
            
        # Save provenance
        with open(out_dir / f"aapl_sample_{name}_provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)

if __name__ == "__main__":
    main()
