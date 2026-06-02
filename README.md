# kg-diagnosis

A diagnostic framework for GraphRAG systems that attributes answer-quality failures to specific knowledge graph (KG) defects.

Given a KG-backed retrieval-augmented generation pipeline, `kg-diagnosis` helps you answer: *which structural or semantic properties of the graph are causing query failures, and which repairs recover the most?*

---

## What it does

- **Controlled degradation** — systematically injects four categories of real-world KG failure (schema decay, entity resolution failure, relational noise, topological fragmentation) at configurable severity levels
- **Feature extraction** — computes 15 subgraph topology and schema-heterogeneity features per query at retrieval time
- **Failure attribution** — trains a LightGBM classifier on those features and uses SHAP to identify which graph properties drive query failure
- **Repair prioritization** — ranks degradation types by the number of failures recovered per fix, enabling targeted remediation

---

## Repository Structure

```
kg-diagnosis/
├── src/
│   ├── graph_diagnostic/          # Core library
│   │   ├── attribution/           # LightGBM + SHAP failure attribution
│   │   ├── corruption/            # Controlled KG degradation simulator
│   │   ├── evaluation/            # Answer quality metrics (F1, exact match)
│   │   ├── features/              # Subgraph feature extractor (15 features)
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
git clone https://github.com/lokeshs-git/kg-diagnosis.git
cd kg-diagnosis

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

Experiments use a **S&P 500 Top-20 subset** of [FinReflectKG](https://huggingface.co/datasets/weaviate/FinReflectKG), a financial knowledge graph. QA pairs are derived from entity-relationship paths in the graph.

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

## Degradation Framework

Four categories of KG failure at three mixed severity levels and three targeted ablations:

| Variant | Schema Decay | Entity Resolution | Relational Noise | Topological Frag. |
|---------|-------------|-------------------|------------------|-------------------|
| `light_mix` | 20.0% | 2.7% | 0.7% | 0.7% |
| `moderate_mix` | 36.6% | 6.7% | 2.3% | 0.7% |
| `heavy_mix` | 53.2% | 10.6% | 4.0% | 1.3% |
| `ablation_schema` | 53.2% | 0% | 0% | 0% |
| `ablation_resolution` | 0% | 10.6% | 0% | 0% |
| `ablation_fragmentation` | 0% | 0% | 0% | 1.3% |

---

## Features

The 15 features extracted per subgraph at retrieval time:

- **Topology:** `node_count`, `edge_count`, `seed_count`, `seed_confidence_mean`, `seed_ambiguity`, `density`, `avg_degree`, `component_count`, `clustering_coeff`, `diameter`, `betweenness_mean`
- **Schema / Heterogeneity:** `property_fill_rate`, `entity_diversity`, `property_diversity`, `relation_diversity`

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
