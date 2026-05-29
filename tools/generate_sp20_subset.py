import logging
import pickle
from pathlib import Path
from graph_diagnostic.data.finreflectkg_loader import load_finreflectkg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Top 20 S&P 500 companies by market cap (approximate for this dataset)
SP20_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
    "META", "BRK-B", "TSLA", "LLY", "V", 
    "UNH", "AVGO", "JPM", "JNJ", "MA", 
    "XOM", "PG", "COST", "HD", "ADBE"
]

def create_sp20_sample():
    """
    Extracts the subgraph for the Top 20 companies.
    Uses pyarrow pushdown filters to keep memory usage low.
    """
    data_dir = Path("data/finreflectkg")
    subsets_dir = Path("data/subsets")
    subsets_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = subsets_dir / "sp20_sample.pkl"
    
    logger.info(f"Extracting sample for {len(SP20_TICKERS)} tickers...")
    
    try:
        # The loader uses pyarrow filters, so it won't load the full 4.1GB file into RAM
        graph = load_finreflectkg(data_dir, tickers=SP20_TICKERS)
        
        logger.info(f"Saving S&P 20 sample to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)
            
        logger.info("Done. You can now run precompute_embeddings.py on this file.")
        
    except Exception as e:
        logger.error(f"Failed to create S&P 20 sample: {e}")

if __name__ == "__main__":
    create_sp20_sample()
