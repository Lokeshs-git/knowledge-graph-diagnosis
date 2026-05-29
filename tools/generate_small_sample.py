import logging
from pathlib import Path
import networkx as nx
import pickle
from graph_diagnostic.data.finreflectkg_loader import load_finreflectkg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_small_sample(ticker: str = "AAPL"):
    """
    Creates a small NetworkX graph for a single ticker to avoid OOM
    and allow for fast development of the QA generation logic.
    """
    data_dir = Path("data/finreflectkg")
    subsets_dir = Path("data/subsets")
    subsets_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = subsets_dir / f"{ticker.lower()}_sample.pkl"
    
    logger.info(f"Extracting small sample for {ticker}...")
    
    try:
        graph = load_finreflectkg(data_dir, tickers=[ticker])
        
        logger.info(f"Saving small sample to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)
            
        logger.info("Done.")
        
    except Exception as e:
        logger.error(f"Failed to create small sample: {e}")

if __name__ == "__main__":
    create_small_sample()
