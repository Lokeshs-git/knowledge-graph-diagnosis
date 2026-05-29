import pytest
import networkx as nx
from graph_diagnostic.data.finreflectkg_loader import load_finreflectkg
from graph_diagnostic.data.subset import extract_top_companies_subset
from graph_diagnostic.data.finreflectkgqa_loader import load_qa_pairs

def test_finreflectkg_loader_smoke(tmp_path):
    """Smoke test to ensure the loader returns a valid Graph object."""
    import pandas as pd
    
    # Create a dummy parquet file
    df = pd.DataFrame({
        "entity": ["NodeA", "NodeB"],
        "target": ["NodeB", "NodeC"],
        "relationship": ["rel1", "rel2"],
        "entity_type": ["Type1", "Type2"],
        "ticker": ["TICK1", "TICK2"],
        "year": [2020, 2021]
    })
    df.to_parquet(tmp_path / "finreflectkg_full.parquet")
    
    graph = load_finreflectkg(tmp_path)
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

def test_subset_extraction_smoke():
    """Smoke test for subset extraction."""
    graph = nx.MultiDiGraph()
    graph.add_node("CompanyA", market_cap=100)
    subset = extract_top_companies_subset(graph, num_companies=1)
    assert isinstance(subset, nx.MultiDiGraph)

def test_qa_pairs_loader_smoke(tmp_path):
    """Smoke test for the QA dataset parser."""
    import json
    
    qa_file = tmp_path / "test_qa.jsonl"
    with open(qa_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question": "Test?", "gold_ids": ["N1"], "metadata": {}}) + "\n")
        
    pairs = load_qa_pairs(qa_file)
    assert len(pairs) == 1
    assert pairs[0][0] == "Test?"
    assert pairs[0][1] == ["N1"]
