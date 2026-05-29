import networkx as nx
import random
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class GraphCorruptor:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(self.random_seed)

    def apply_degradation(self, graph: nx.MultiDiGraph, rates: Dict[str, float]) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
        """
        Applies multiple degradation rules to the graph simultaneously based on the provided rates.
        Returns the corrupted graph and a provenance dictionary logging the modifications.
        """
        # Create a deep copy to avoid modifying the original
        corrupted = graph.copy()
        provenance = {
            "missing_properties": [],
            "duplicate_entities": [],
            "flipped_relations": [],
            "deleted_edges": []
        }
        
        logger.info(f"Applying degradation with rates: {rates}")
        
        # 1. Missing Properties (drop properties from nodes)
        missing_rate = rates.get("missing_properties", 0.0)
        if missing_rate > 0:
            for node, data in corrupted.nodes(data=True):
                # If we had multiple properties, we'd randomly drop them.
                # For this dataset, we'll just rename 'type' to 'unknown' to simulate lost schema
                if random.random() < missing_rate:
                    if 'type' in data and data['type'] != 'unknown':
                        provenance["missing_properties"].append(node)
                        data['type'] = 'unknown'

        # 2. Duplicate Entities (clone nodes and split edges)
        duplicate_rate = rates.get("duplicate_entities", 0.0)
        if duplicate_rate > 0:
            nodes_to_duplicate = [n for n in corrupted.nodes() if random.random() < duplicate_rate]
            for node in nodes_to_duplicate:
                clone_name = f"{node}_duplicate"
                corrupted.add_node(clone_name, **corrupted.nodes[node])
                provenance["duplicate_entities"].append((node, clone_name))
                
                # Redistribute edges (about 50% go to the clone, 50% stay on original)
                edges_to_move = []
                for u, v, key, data in corrupted.edges(node, keys=True, data=True):
                    if random.random() < 0.5:
                        edges_to_move.append((u, v, key, data))
                
                for u, v, key, data in edges_to_move:
                    if u == node:
                        corrupted.add_edge(clone_name, v, **data)
                    else:
                        corrupted.add_edge(u, clone_name, **data)
                    corrupted.remove_edge(u, v, key=key)

        # 3. Flipped/Mislabeled Relations
        flip_rate = rates.get("flipped_relations", 0.0)
        if flip_rate > 0:
            edges_to_flip = []
            for u, v, key, data in corrupted.edges(keys=True, data=True):
                if random.random() < flip_rate:
                    edges_to_flip.append((u, v, key, data))
                    
            for u, v, key, data in edges_to_flip:
                # Add flipped edge
                corrupted.add_edge(v, u, **data)
                # Remove original
                corrupted.remove_edge(u, v, key=key)
                provenance["flipped_relations"].append((u, v, key))

        # 4. Orphan Subgraphs (delete bridging edges)
        orphan_rate = rates.get("orphan_subgraphs", 0.0)
        if orphan_rate > 0:
            edges_to_delete = []
            for u, v, key in corrupted.edges(keys=True):
                if random.random() < orphan_rate:
                    edges_to_delete.append((u, v, key))
                    
            for u, v, key in edges_to_delete:
                corrupted.remove_edge(u, v, key=key)
                provenance["deleted_edges"].append((u, v, key))

        logger.info(f"Corrupted graph stats: {corrupted.number_of_nodes()} nodes, {corrupted.number_of_edges()} edges")
        return corrupted, provenance
