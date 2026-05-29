import networkx as nx
import numpy as np
import math
from collections import Counter
from typing import Dict, Any, List, Tuple

class FeatureExtractor:
    """
    Advanced Feature Extractor for Scientific Graph Diagnostic.
    Maps local subgraph topology to answer-quality predictors.
    """
    
    @staticmethod
    def _calculate_entropy(labels: List[str]) -> float:
        """Calculates Shannon entropy for a list of categorical labels."""
        if not labels:
            return 0.0
        counts = Counter(labels)
        probabilities = [count / len(labels) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    @staticmethod
    def extract_all(subgraph: nx.MultiDiGraph, seeds_with_scores: List[Tuple[str, float]]) -> Dict[str, float]:
        features = {}
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        
        # Base features that always exist
        features["node_count"] = float(n_nodes)
        features["edge_count"] = float(n_edges)
        features["seed_count"] = float(len(seeds_with_scores))

        # 1. Retrieval Semantic Signals
        if seeds_with_scores:
            scores = [score for _, score in seeds_with_scores]
            features["seed_confidence_mean"] = float(np.mean(scores))
            features["seed_ambiguity"] = float(np.std(scores))
        else:
            features["seed_confidence_mean"] = 0.0
            features["seed_ambiguity"] = 0.0

        if n_nodes < 2:
            return {
                **features,
                "density": 0.0, 
                "avg_degree": 0.0, 
                "component_count": 1.0 if n_nodes == 1 else 0.0, 
                "clustering_coeff": 0.0, 
                "diameter": 0.0, 
                "betweenness_mean": 0.0, 
                "property_fill_rate": 0.0,
                "entity_diversity": 0.0,
                "relation_diversity": 0.0
            }

        # 2. Connectivity & Fragmentation
        features["density"] = n_edges / (n_nodes * (n_nodes - 1))
        features["avg_degree"] = float(sum(dict(subgraph.degree()).values()) / n_nodes)
        
        # Convert to simple undirected graph for topology metrics
        # (Collapses multiple edges between nodes into a single edge)
        simple_undirected = nx.Graph(subgraph)
        features["component_count"] = float(nx.number_connected_components(simple_undirected))
        
        # 3. Topology (Advanced Metrics) - Capped for performance
        if n_nodes < 1000:
            features["clustering_coeff"] = float(nx.average_clustering(simple_undirected))
            
            try:
                largest_cc_nodes = max(nx.connected_components(simple_undirected), key=len)
                largest_cc = simple_undirected.subgraph(largest_cc_nodes)
                features["diameter"] = float(nx.diameter(largest_cc))
            except Exception:
                features["diameter"] = 0.0
                
            # We sample k nodes to approximate centrality
            k_sample = min(100, n_nodes)
            centrality = nx.betweenness_centrality(simple_undirected, k=k_sample)
            features["betweenness_mean"] = float(np.mean(list(centrality.values())))
        else:
            # If the subgraph is massive, we assume it is a giant hairball. 
            # Calculating these exactly will deadlock the CPU.
            features["clustering_coeff"] = 0.0
            features["diameter"] = -1.0 # Indicator of massive graph
            features["betweenness_mean"] = 0.0

        # 4. Schema Integrity & Heterogeneity (Domain Agnostic)
        filled = 0
        node_types = []
        property_keys = []
        
        for _, d in subgraph.nodes(data=True):
            n_type = str(d.get('type', 'unknown'))
            node_types.append(n_type)
            
            # Track all property keys for diversity calculation
            for key, val in d.items():
                if val != 'unknown' and val is not None:
                    property_keys.append(key)
            
            if d.get('ticker') and d.get('ticker') != 'unknown':
                filled += 1
            if n_type != 'unknown':
                filled += 1
                
        features["property_fill_rate"] = filled / (n_nodes * 2) if n_nodes > 0 else 0.0
        
        # Calculate Heterogeneity using Shannon Entropy
        features["entity_diversity"] = FeatureExtractor._calculate_entropy(node_types)
        features["property_diversity"] = FeatureExtractor._calculate_entropy(property_keys)
        
        edge_types = [str(d.get('type', key)) for _, _, key, d in subgraph.edges(keys=True, data=True)]
        features["relation_diversity"] = FeatureExtractor._calculate_entropy(edge_types)

        return features
