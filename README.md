# kg-diagnosis

An ML-based diagnostic framework for assessing knowledge graph (KG) structural quality in GraphRAG systems.

Given a KG-backed retrieval-augmented generation pipeline, `kg-diagnosis` identifies *which structural properties of the graph are causing query failures* and produces an engineering priority table ranking maintenance actions by failures recovered per 1,000 queries.

The framework operates on structural primitives of any property graph. The source of degradation — data drift, incomplete updates, schema evolution, or merge errors — is immaterial to the diagnostic pipeline.

---

## Framework

The framework comprises three composable components applied sequentially:

**1. Subgraph feature extraction**
For each question and graph state, the retrieval module produces a subgraph. The framework extracts 15 structural features organized into three groups:

| Group | Features |
|-------|---------|
| Topology | `node_count`, `edge_count`, `density`, `avg_degree`, `component_count`, `clustering_coeff`, `betweenness_mean`, `diameter` |
| Retrieval | `seed_count`, `seed_confidence_mean`, `seed_ambiguity`, `property_fill_rate` |
| Diversity | `entity_diversity`, `relation_diversity`, `property_diversity` |

**2. Delta-formulation attribution**
Rather than training on raw feature values — which conflates question difficulty with structural signal — the framework computes per-question feature deltas between a target KG state and a clean reference baseline. This removes the question-difficulty confound: each row in the resulting delta dataset measures the *marginal* structural change and its associated performance change.

**3. Binary failure classification and engineering priority table**
A LightGBM classifier trained on the delta dataset predicts whether a degraded variant causes a previously answerable question to fail completely (F1 = 0). SHAP attribution decomposes predictions into per-feature contributions. The terminal artifact is an **engineering priority table** mapping structural dimensions to failures recovered per 1,000 queries — a ranked maintenance roadmap for KG teams.

---

## Validation via controlled degradation

To validate the framework's discriminative power, the repository includes a controlled degradation simulator that injects four categories of real-world KG failure at configurable severity levels:

| Failure Mode | Description |
|-------------|-------------|
| Schema / attribute decay | Drop node properties with probability *r* |
| Entity resolution failure | Create synthetic duplicate nodes with redistributed edges |
| Semantic / relational noise | Reverse edge direction for fraction *r* of edges (negative control) |
| Topological fragmentation | Delete fraction *r* of edges |

Operators are seeded for reproducibility and composable: any combination can be applied in sequence with recorded severities.

---

## Repository Structure

```
kg-diagnosis/
├── src/
│   ├── graph_diagnostic/          # Core library
│   │   ├── attribution/           # LightGBM + SHAP failure classification
│   │   ├── corruption/            # Controlled KG degradation simulator
│   │   ├── evaluation/            # Answer quality metrics (F1, exact match)
│   │   ├── features/              # 15-feature subgraph extractor
│   │   └── pipeline/              # Retrieval and generation pipeline
│   └── quickstart/                # LLMClient: unified Anthropic/OpenRouter/Gemini interface
│
├── experiments/
│   └── exp2_1000qa_7variants/
│       ├── config.yaml            # Experiment specification (degradation rates, model params)
│       └── exp2_analysis.ipynb    # Feature distributions, classifier, SHAP attribution
│   ├── phase3a_retrieval.py       # Stage 1: retrieval cache (subgraph + features)
│   ├── phase3b_generation.py      # Stage 2: parallel LLM answer generation
│   ├── phase3b_resume.py          # Resume partial generation runs
│   └── phase3_data_generation.py  # Combined pipeline runner
│
├── evals/
│   ├── runner.py                  # Evaluation harness
│   ├── scorers.py                 # F1 and answer scoring
│   └── run_example.py             # Standalone eval example
│
├── notebooks/
│   ├── 01_exploration.ipynb       # Dataset and graph exploration
│   └── 02_attribution_analysis.ipynb  # SHAP attribution deep-dive
│
├── tests/                         # Unit tests
├── tools/                         # Data preparation and reproducibility scripts
├── data/                          # Data directory (not tracked — see Data section)
├── pyproject.toml                 # Dependencies and project config
└── Makefile                       # Common tasks
```

---

## Requirements

- **Python 3.12** (see `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **API keys:**
  - `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` — for LLM answer generation
  - `GEMINI_API_KEY` — for QA pair generation

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Lokeshs-git/knowledge-graph-diagnosis.git
cd knowledge-graph-diagnosis

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 3. Install dependencies
uv sync

# 4. Run tests to verify setup
uv run pytest tests/

# 5. Prepare data (see Data section below)
uv run python tools/download_data.py
uv run python tools/generate_sp20_subset.py
uv run python tools/generate_degraded_graphs.py

# 6. Run the experiment pipeline
uv run python experiments/phase3a_retrieval.py   # retrieval + feature extraction
uv run python experiments/phase3b_generation.py  # LLM answer generation

# 7. Analyze results
jupyter lab experiments/exp2_1000qa_7variants/exp2_analysis.ipynb
```

---

## Data

The included case study uses a **S&P 500 Top-20 subset** of [FinReflectKG](https://huggingface.co/datasets/weaviate/FinReflectKG), a financial knowledge graph. QA pairs are derived from 2-hop entity-relationship paths in the graph.

Data files are **not included** in this repository due to size. The `tools/` directory contains scripts to regenerate all data artifacts:

| Script | Purpose |
|--------|---------|
| `tools/download_data.py` | Download FinReflectKG from HuggingFace Hub |
| `tools/generate_sp20_subset.py` | Extract S&P 500 Top-20 subgraph |
| `tools/generate_degraded_graphs.py` | Apply controlled degradation variants |
| `tools/generate_small_sample.py` | Generate a small sample for smoke testing |
| `tools/precompute_embeddings.py` | Precompute sentence embeddings for retrieval |
| `tools/validate_qa.py` | Validate QA pair quality |
| `tools/generate_additional_qa.py` | Generate additional QA pairs with Gemini |
| `tools/generate_sp20_degraded.py` | Apply degradation specifically to SP20 subset |

After running these scripts, your `data/` directory should contain:

```
data/
├── finreflectkgqa/
│   └── production_exp2_1000.jsonl   # Validated QA pairs
└── subsets/
    ├── sp20_sample.pkl              # Clean SP20 knowledge graph
    └── degraded_exp2/               # Degraded variants
        ├── light_mix.pkl
        ├── moderate_mix.pkl
        ├── heavy_mix.pkl
        ├── ablation_schema.pkl
        ├── ablation_resolution.pkl
        └── ablation_fragmentation.pkl
```

---

## Citing this work

If you use this framework in your research, please cite the relevant paper(s) listed in [`CITATION.cff`](CITATION.cff).

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Dependencies

Key dependencies (see `pyproject.toml` for the full list):

- `lightgbm` — gradient-boosted failure classifier
- `shap` — Shapley value attribution
- `networkx` — graph construction and feature extraction
- `sentence-transformers` — embedding-based retrieval
- `anthropic`, `openai`, `google-generativeai` — LLM providers
- `pandas`, `numpy`, `scikit-learn` — data processing and evaluation
