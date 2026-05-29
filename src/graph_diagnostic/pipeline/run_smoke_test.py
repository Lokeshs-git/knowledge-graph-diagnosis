import logging
import pickle
import json
from pathlib import Path
from graph_diagnostic.pipeline.retrieval import GraphRetriever
from graph_diagnostic.pipeline.generation import GraphGenerator
from graph_diagnostic.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test():
    """Runs the 5 synthetic questions against the AAPL subgraph."""
    graph_path = Path("data/subsets/aapl_sample.pkl")
    qa_path = Path("data/finreflectkgqa/small_sample_qa.jsonl")
    
    if not graph_path.exists() or not qa_path.exists():
        logger.error("Missing data. Run Phase 0 extraction first.")
        return
        
    logger.info("Loading graph...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
        
    logger.info("Initializing Pipeline...")
    retriever = GraphRetriever(graph)
    generator = GraphGenerator()
    
    logger.info("Loading QA pairs...")
    with open(qa_path, "r") as f:
        qa_pairs = [json.loads(line) for line in f]
        
    results = []
    
    for i, qa in enumerate(qa_pairs):
        query = qa["question"]
        gold_ids = set(qa["gold_ids"])
        
        logger.info(f"\n--- Query {i+1}/{len(qa_pairs)} ---")
        logger.info(f"Q: {query}")
        
        # 1. Retrieve
        subgraph, seeds = retriever.retrieve_subgraph(query, k_seeds=2, hops=2)
        logger.info(f"Retrieved Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        # 2. Generate
        predicted_ids_list = generator.generate_answer(query, subgraph)
        predicted_ids = set(predicted_ids_list)
        
        # 3. Evaluate
        metrics = compute_metrics(predicted_ids, gold_ids)
        
        logger.info(f"Gold: {gold_ids}")
        logger.info(f"Predicted: {predicted_ids}")
        logger.info(f"F1 Score: {metrics['f1']:.2f}")
        
        results.append({
            "query": query,
            "f1": metrics["f1"]
        })
        
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    logger.info(f"\nOverall Pipeline F1 Score: {avg_f1:.2f}")

if __name__ == "__main__":
    run_smoke_test()
