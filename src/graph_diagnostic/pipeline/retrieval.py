import json
import logging
import os
from pathlib import Path

import networkx as nx
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from quickstart.llm import LLMClient

logger = logging.getLogger(__name__)

_KW_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")


class GraphRetriever:
    def __init__(self, graph: nx.MultiDiGraph, embeddings_path: str | Path, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the retriever with a NetworkX graph.
        Loads precomputed embeddings from the specified path.
        """
        self.graph = graph
        self.node_ids = list(graph.nodes())
        # Map node_id to its index in the embedding array for fast lookup
        self.node_to_idx = {node: i for i, node in enumerate(self.node_ids)}

        emb_path = Path(embeddings_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not found at {emb_path}. Run tools/precompute_embeddings.py")

        logger.info(f"Loading precomputed embeddings from {emb_path}...")
        self.node_embeddings = np.load(emb_path)

        logger.info(f"Loading query encoder model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        provider = os.environ.get("GENERATION_PROVIDER", "anthropic").lower()
        if provider == "openrouter":
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            self._llm_complete = self._complete_openrouter
            logger.info(f"Retriever keyword extraction using OpenRouter ({_KW_MODEL})")
        else:
            self._anthropic_client = LLMClient()
            self._llm_complete = self._complete_anthropic
            logger.info("Retriever keyword extraction using Anthropic LLMClient")

        # Edge profile → embedding cache, populated lazily during walks.
        # Shared across all questions on the same graph; warm after ~1 question.
        self._edge_emb_cache: dict[str, np.ndarray] = {}

        logger.info(f"Retriever initialized with {graph.number_of_nodes()} nodes (using cached embeddings).")

    def _complete_openrouter(self, prompt: str) -> str:
        response = self._openrouter_client.chat.completions.create(
            model=_KW_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _complete_anthropic(self, prompt: str) -> str:
        return self._anthropic_client.complete(prompt, temperature=0.0, response_mime_type="application/json")

    def _extract_keywords(self, query: str) -> tuple[list[str], str]:
        """Extracts semantic keywords and an answer description from the query.

        Returns a tuple of (keywords, answer_description). The answer_description
        is a free-text phrase describing what the answer entity should look like,
        used as an additional semantic anchor in seed selection.
        """
        prompt = f"""You are an expert phrase extraction system.

Your task is to extract semantic phrases from a natural language question AND describe what the answer entity should look like.

Instructions:
1. Extract meaningful phrases exactly as they appear in the question whenever possible.
2. Include noun phrases, verb phrases, relational phrases, causal phrases, and intent-bearing phrases.
3. Preserve multi-word phrases. Avoid generic stop words unless part of a meaningful phrase.
4. Do NOT summarize, infer, or paraphrase the keywords.
5. For "answer_description": write a short phrase (5-15 words) describing the type and role of the expected answer entity, based on what the question is asking for.

Query: {query}

Return ONLY a JSON object with two keys:
- "keywords": list of extracted phrases
- "answer_description": short description of the expected answer entity

Example: {{"keywords": ["consumer confidence", "fuel and energy costs"], "answer_description": "an individual who holds a financial stake in a company"}}
"""
        try:
            response_text = self._llm_complete(prompt)
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            data = json.loads(response_text[start_idx:end_idx])
            keywords = data.get("keywords", [])
            answer_description = data.get("answer_description", "")
            if not keywords:
                return [query], answer_description
            return keywords, answer_description
        except Exception as e:
            logger.warning(f"Keyword extraction failed. Error: {e}")
            return [query], ""

    def get_seed_nodes(
        self,
        query: str,
        k_per_keyword: int = 3,
        max_total: int = 10,
        keywords_override: tuple[list[str], str] | None = None,
    ) -> list[tuple[str, float]]:
        """Round-Robin Seed Union with answer-description anchor.

        The answer_description extracted from the query is used as an additional
        high-priority keyword so that seed selection is biased toward nodes matching
        the expected answer entity, not just intermediate concepts.

        Pass keywords_override=(keywords, answer_description) to skip LLM extraction.
        """
        if keywords_override is not None:
            keywords, answer_description = keywords_override
        else:
            keywords, answer_description = self._extract_keywords(query)
        logger.info(f"  [Retriever Trace] Extracted Keywords: {keywords}")
        logger.info(f"  [Retriever Trace] Answer Description: {answer_description}")

        seeds_per_kw = {}
        # Full query as fallback anchor; answer_description as forward anchor
        all_kws = [*keywords, query]
        if answer_description:
            all_kws = [answer_description, *all_kws]

        for kw in all_kws:
            kw_embedding = self.model.encode([kw], show_progress_bar=False)
            sims = cosine_similarity(kw_embedding, self.node_embeddings)[0]

            top_indices = np.argsort(sims)[-k_per_keyword:][::-1]
            seeds_per_kw[kw] = [(self.node_ids[i], float(sims[i])) for i in top_indices]

        # Union across keywords, dedupe by max score;
        # answer_description seeds get a 0.1 boost so they survive the max_total cap
        combined_seeds = {}
        answer_seeds = {node for node, _ in seeds_per_kw.get(answer_description, [])} if answer_description else set()
        for _kw, seeds in seeds_per_kw.items():
            for node, score in seeds:
                boosted = score + 0.10 if node in answer_seeds else score
                combined_seeds[node] = max(combined_seeds.get(node, 0.0), boosted)

        sorted_seeds = sorted(combined_seeds.items(), key=lambda x: x[1], reverse=True)[:max_total]
        return sorted_seeds, all_kws

    def _format_edge_profile(self, u: str, v: str, key: str, is_outgoing: bool) -> str:
        """Minor Fix: Clean edge formatting to avoid embedding dilution"""
        target_node = v if is_outgoing else u
        t_type = self.graph.nodes[target_node].get('type', 'unknown')
        edge_type = self.graph.edges[u, v, key].get('type', key)

        # Clean, natural language string
        if t_type == 'unknown':
            return f"{edge_type} {target_node}"
        return f"{edge_type} {target_node}, a {t_type}"

    def retrieve_subgraph(
        self,
        query: str,
        k_seeds: int = 3,
        hops: int = 2,
        max_edges: int = 250,
        keywords_override: tuple[list[str], str] | None = None,
    ) -> tuple[nx.MultiDiGraph, list[tuple[str, float]]]:
        """
        SOTA Semantic Retriever implementing Keyword Max-Scoring and Decaying Budgets.

        Pass keywords_override=(keywords, answer_description) to skip LLM keyword extraction.
        """
        # 1. Round-Robin Anchoring; max_total scales with k_seeds to avoid capping breadth gains
        seeds_with_scores, extracted_kws = self.get_seed_nodes(
            query, k_per_keyword=k_seeds, max_total=k_seeds * 5, keywords_override=keywords_override
        )
        top_seeds = [node for node, score in seeds_with_scores]
        logger.info(f"  [Retriever Trace] Top {len(top_seeds)} Starting Seeds: {top_seeds}")

        # 2. Embed all keywords for Issue 3 (Keyword Max-Scoring)
        kw_embeddings = self.model.encode(extracted_kws, show_progress_bar=False)

        subgraph = nx.MultiDiGraph()
        for seed in top_seeds:
            subgraph.add_node(seed, **self.graph.nodes[seed])

        frontier = list(top_seeds)
        edges_added = 0
        visited_edges = set()

        # Decaying hop budget: front-load edges at hop 1 (highest signal), taper toward terminal hops
        if hops == 2:
            hop_budgets = [int(max_edges * 0.6), int(max_edges * 0.4)]
        elif hops == 3:
            hop_budgets = [int(max_edges * 0.5), int(max_edges * 0.3), int(max_edges * 0.2)]
        else:
            hop_budgets = [int(max_edges / hops)] * hops

        for hop in range(hops):
            budget_this_hop = hop_budgets[hop] if hop < len(hop_budgets) else int(max_edges / hops)
            edges_this_hop = 0

            if edges_added >= max_edges or not frontier:
                break

            candidate_edges = []
            edge_profiles = []

            for node in frontier:
                for u, v, key in self.graph.out_edges(node, keys=True):
                    if (u, v, key) not in visited_edges:
                        profile = self._format_edge_profile(u, v, key, is_outgoing=True)
                        candidate_edges.append((u, v, key, v))
                        edge_profiles.append(profile)

                for u, v, key in self.graph.in_edges(node, keys=True):
                    if (u, v, key) not in visited_edges:
                        profile = self._format_edge_profile(u, v, key, is_outgoing=False)
                        candidate_edges.append((u, v, key, u))
                        edge_profiles.append(profile)

            if not candidate_edges:
                break

            # Encode only profiles not yet seen; reuse cache for the rest.
            # After question 1 the cache covers most of the graph — subsequent
            # questions cost only the np.array lookup, not re-encoding.
            new_profiles = [p for p in edge_profiles if p not in self._edge_emb_cache]
            if new_profiles:
                new_embs = self.model.encode(new_profiles, show_progress_bar=False, batch_size=512)
                for p, emb in zip(new_profiles, new_embs, strict=True):
                    self._edge_emb_cache[p] = emb
            edge_embs = np.array([self._edge_emb_cache[p] for p in edge_profiles])

            # Issue 3 Fix: Keyword Max-Scoring (score edge against all keywords, take the max)
            # sims shape: (len(keywords), len(edges))
            sims_matrix = cosine_similarity(kw_embeddings, edge_embs)
            # max across keywords (axis 0) -> shape: (len(edges),)
            max_sims = np.max(sims_matrix, axis=0)

            ranked_indices = np.argsort(max_sims)[::-1]
            next_frontier = set()

            for idx in ranked_indices:
                if edges_added >= max_edges or edges_this_hop >= budget_this_hop:
                    break

                u, v, key, target_node = candidate_edges[idx]
                score = max_sims[idx]

                # Hop-adaptive threshold: strict at hop 1 to avoid noisy expansion,
                # permissive at hop 2+ because answer-terminal edges score lower by design.
                threshold = 0.30 if hop == 0 else 0.15
                if score < threshold:
                    continue

                visited_edges.add((u, v, key))

                if not subgraph.has_node(u):
                    subgraph.add_node(u, **self.graph.nodes[u])
                if not subgraph.has_node(v):
                    subgraph.add_node(v, **self.graph.nodes[v])

                subgraph.add_edge(u, v, key=key, **self.graph.edges[u, v, key])

                next_frontier.add(target_node)
                edges_added += 1
                edges_this_hop += 1

            frontier = list(next_frontier)

        logger.info(f"  [Retriever Trace] Edge-Scored Walk complete. Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        return subgraph, seeds_with_scores
