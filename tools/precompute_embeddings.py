import logging
import pickle
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute(graph_path: str, out_path: str):
    graph_path = Path(graph_path)
    out_path = Path(out_path)
    
    if not graph_path.exists():
        logger.error(f"Graph not found at {graph_path}")
        return

    logger.info(f"Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
        
    nodes = list(graph.nodes())
    logger.info(f"Loaded {len(nodes)} nodes.")
    
    logger.info("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("Computing embeddings (this may take a minute but only happens once)...")
    # Build 'Node Profiles' combining name and metadata for richer semantic matching
    texts = []
    for node, data in graph.nodes(data=True):
        n_type = data.get('type', 'unknown')
        n_ticker = data.get('ticker', 'unknown')
        edge_types = set()
        for _, _, d in graph.out_edges(node, data=True):
            if 'type' in d:
                edge_types.add(d['type'])
        for _, _, d in graph.in_edges(node, data=True):
            if 'type' in d:
                edge_types.add(d['type'])
        roles = ', '.join(list(edge_types)[:4]) if edge_types else 'unknown'
        profile = f"{node} ({n_type}) [Ticker: {n_ticker}] [Roles: {roles}]"
        texts.append(profile)
    
    # Use a larger batch size on GPU
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    
    logger.info(f"Saving embeddings to {out_path}...")
    np.save(out_path, embeddings)
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="data/subsets/aapl_sample.pkl")
    parser.add_argument("--output", default="data/subsets/aapl_sample_embeddings.npy")
    args = parser.parse_args()
    
    precompute(args.graph, args.output)
