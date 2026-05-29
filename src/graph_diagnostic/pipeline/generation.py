import json
import logging
import os

import networkx as nx
from openai import OpenAI

from quickstart import LLMClient

logger = logging.getLogger(__name__)

_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")


class GraphGenerator:
    def __init__(self):
        provider = os.environ.get("GENERATION_PROVIDER", "anthropic").lower()
        if provider == "openrouter":
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            self._call = self._call_openrouter
            logger.info(f"Generator using OpenRouter ({_OPENROUTER_MODEL})")
        else:
            self._anthropic_client = LLMClient()
            self._call = self._call_anthropic
            logger.info("Generator using Anthropic LLMClient")

    def _call_openrouter(self, prompt: str) -> str:
        response = self._openrouter_client.chat.completions.create(
            model=_OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        return self._anthropic_client.complete(prompt, temperature=0.0)

    @staticmethod
    def format_subgraph_context(subgraph: nx.MultiDiGraph) -> str:
        """Converts the NetworkX subgraph into a string representation for the LLM."""
        lines = []
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            u_type = subgraph.nodes[u].get('type', 'Unknown')
            v_type = subgraph.nodes[v].get('type', 'Unknown')
            rel_type = data.get('type', key)
            lines.append(f"({u} [{u_type}]) --[{rel_type}]--> ({v} [{v_type}])")
        return "\n".join(lines)

    def generate_answer(self, query: str, subgraph: nx.MultiDiGraph) -> list[str]:
        """
        Takes a user query and a retrieved subgraph context, asks the LLM to answer the
        query using ONLY the context, and returns a list of Node IDs as the answer.
        """
        context_str = GraphGenerator.format_subgraph_context(subgraph)
        return self.generate_from_context(query, context_str)

    def generate_from_context(self, query: str, context_str: str) -> list[str]:
        """Generates answer IDs from a pre-formatted context string.

        Accepts the subgraph already serialized to text, enabling reuse across
        multiple calls without re-formatting the same subgraph.
        """
        prompt = f"""You are a GraphRAG entity extraction system.
You will be provided with a user query and a Knowledge Graph Context represented as (Source Node) --[Relationship]--> (Target Node).

Your job is to read the context and return the exact Node ID(s) that answer the question.

Knowledge Graph Context:
{context_str}

User Query: {query}

Instructions:
1. Answer ONLY using the entities provided in the Knowledge Graph Context.
2. If the answer is a number or percentage, look for a node that matches that value exactly (e.g. "10 %").
3. Do not write full sentences. Your answer must be a JSON object with a single key "predicted_ids" containing a list of strings.
4. The strings MUST MATCH the entity names as they appear inside the parentheses in the context.

Example Response:
{{
  "predicted_ids": ["Entity A", "10 %"]
}}
"""

        try:
            response_text = self._call(prompt)

            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.warning(f"Failed to parse JSON from LLM response: {response_text}")
                return []

            data = json.loads(response_text[start_idx:end_idx])
            return data.get("predicted_ids", [])

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return []
