# Diagnosing Knowledge Graph Quality in GraphRAG: A Controlled Degradation and SHAP Attribution Framework

> **Paper:** "Diagnosing Knowledge Graph Quality in GraphRAG: A Controlled Degradation and SHAP Attribution Framework"
> *IEEE Transactions on Knowledge and Data Engineering* (Under Review, 2025)

---

## Abstract

Graph Retrieval-Augmented Generation (GraphRAG) systems depend on the structural and semantic integrity of an underlying Knowledge Graph (KG). We present a controlled degradation framework that systematically introduces four categories of real-world KG failure — schema decay, entity resolution failure, relational noise, and topological fragmentation — across six severity variants applied to a financial knowledge graph (~129,000 nodes). For 1,000 graph-grounded questions evaluated across all variants (7,000 total evaluations), we extract 15 subgraph topology and schema-heterogeneity features and use SHAP attribution on a LightGBM failure classifier (AUC = 0.794) to identify root causes. Heavy degradation increases query failure rate from 16.6% to 27.0% (+104 complete failures per 1,000 queries). Entity resolution repair recovers the most failures (15.3 per 1,000 queries), inverting the intuition that topology dominates.

---

## Key Findings

- **AUC = 0.794** binary failure classifier predicts complete query failure from structural graph features alone
- **Heavy degradation:** 16.6% → 27.0% failure rate (+104 complete failures per 1,000 queries)
- **`seed_ambiguity` is the top SHAP predictor** (12.1% of total attribution) — entry-point health dominates topology
- **Each failure mode produces a distinct SHAP fingerprint**, enabling root-cause isolation without re-running the pipeline
- **Entity resolution repair recovers the most failures** (15.3 per 1,000 queries), outperforming topology-focused repairs

---

## Repository Structure

```
kg-diagnosis/
├── src/
│   ├── graph_diagnostic/          # Main research library
│   │   ├── attribution/           # LightGBM + SHAP failure attribution
│   │   ├── corruption/            # Controlled KG degradation simulator
│   │   ├── evaluation/            # Answer quality metrics (F1, exact match)
│   │   ├── features/              # 15-feature subgraph extractor
│   │   └── pipeline/              # Retrieval and generation pipeline
│   └── quickstart/                # LLMClient: unified Anthropic/OpenRouter/Gemini interface
│
├── experiments/
│   └── exp2_1000qa_7variants/
│       ├── config.yaml            # Full experiment specification (degradation rates, model params)
│       └── exp2_analysis.ipynb    # Main analysis notebook: features, classifier, SHAP
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
  - `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` — for LLM answer generation (DeepSeek via OpenRouter recommended)
  - `GEMINI_API_KEY` — for QA pair generation (Gemini 2.5 Pro)

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/[ANONYMIZED]/kg-diagnosis.git
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

The experiment uses a **S&P 500 Top-20 subset** of [FinReflectKG](https://huggingface.co/datasets/weaviate/FinReflectKG), a financial knowledge graph (~129,000 nodes in the full graph, ~15,000 nodes in the SP20 subset). QA pairs are derived from entity-relationship paths in the graph.

Data files are **not included** in this repository due to size. The `tools/` directory contains scripts to regenerate all data artifacts:

| Script | Purpose |
|--------|---------|
| `tools/download_data.py` | Download FinReflectKG from HuggingFace Hub |
| `tools/generate_sp20_subset.py` | Extract S&P 500 Top-20 subgraph |
| `tools/generate_degraded_graphs.py` | Apply controlled degradation (6 variants) |
| `tools/generate_small_sample.py` | Generate a small sample for smoke testing |
| `tools/precompute_embeddings.py` | Precompute sentence embeddings for retrieval |
| `tools/validate_qa.py` | Validate QA pair quality |
| `tools/generate_additional_qa.py` | Generate additional QA pairs with Gemini |
| `tools/generate_sp20_degraded.py` | Apply degradation specifically to SP20 subset |

After running these scripts, your `data/` directory should contain:

```
data/
├── finreflectkgqa/
│   └── production_exp2_1000.jsonl   # 1,000 validated QA pairs
└── subsets/
    ├── sp20_sample.pkl              # Clean SP20 knowledge graph
    └── degraded_exp2/               # 6 degraded variants
        ├── light_mix.pkl
        ├── moderate_mix.pkl
        ├── heavy_mix.pkl
        ├── ablation_schema.pkl
        ├── ablation_resolution.pkl
        └── ablation_fragmentation.pkl
```

---

## Degradation Framework

The experiment applies four categories of real-world KG failure at three mixed severity levels and three targeted ablations:

| Variant | Schema Decay | Entity Resolution | Relational Noise | Topological Frag. |
|---------|-------------|-------------------|------------------|-------------------|
| `light_mix` | 20.0% | 2.7% | 0.7% | 0.7% |
| `moderate_mix` | 36.6% | 6.7% | 2.3% | 0.7% |
| `heavy_mix` | 53.2% | 10.6% | 4.0% | 1.3% |
| `ablation_schema` | 53.2% | 0% | 0% | 0% |
| `ablation_resolution` | 0% | 10.6% | 0% | 0% |
| `ablation_fragmentation` | 0% | 0% | 0% | 1.3% |

---

## Reproducing the Analysis

The main analysis notebook `experiments/exp2_1000qa_7variants/exp2_analysis.ipynb` covers:

1. Feature distribution across degradation variants
2. LightGBM binary failure classifier training and evaluation (AUC = 0.794)
3. SHAP global attribution — which features drive failure prediction
4. Per-variant SHAP fingerprints for root-cause isolation
5. Repair prioritization: which degradation type to fix first for maximum recovery

See `docs/methodology/FEATURE_GUIDE.md` (internal) for the full feature definitions. The 15 features are:

- **Topology:** `node_count`, `edge_count`, `seed_count`, `seed_confidence_mean`, `seed_ambiguity`, `density`, `avg_degree`, `component_count`, `clustering_coeff`, `diameter`, `betweenness_mean`
- **Schema/Heterogeneity:** `property_fill_rate`, `entity_diversity`, `property_diversity`, `relation_diversity`

---

## Citation

```bibtex
@article{anonymized2025kgdiagnosis,
  author    = {[ANONYMIZED]},
  title     = {Diagnosing Knowledge Graph Quality in {GraphRAG}: A Controlled
               Degradation and {SHAP} Attribution Framework},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  year      = {2025},
  note      = {Under review}
}
```

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
