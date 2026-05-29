"""End-to-end eval example.

Runs the capitals dataset through Claude, scores with `contains` (lenient
since the model may add extra words), prints a table, and writes results
to `evals/results/`.

Run with:
    uv run python -m evals.run_example
"""

from datetime import datetime
from pathlib import Path

from quickstart.llm import LLMClient
from quickstart.logging import setup_logging

from evals.runner import load_dataset, run_eval, save_results, summarize
from evals.scorers import contains, exact_match


def main() -> None:
    setup_logging()

    # 1. Load dataset
    dataset_path = Path(__file__).parent / "datasets" / "capitals.jsonl"
    examples = load_dataset(dataset_path)

    # 2. Define the unit under test — here, a single Claude call with
    #    a system prompt that asks for terse answers
    client = LLMClient(
        system="Answer in one word only. No punctuation, no explanation.",
    )

    def model_fn(prompt: str) -> str:
        return client.complete(prompt, max_tokens=20, temperature=0.0)

    # 3. Pick scorers — start with cheap, deterministic ones.
    #    Add LLMJudge for open-ended outputs.
    scorers = {
        "exact": exact_match,
        "contains": contains,
    }

    # 4. Run + report
    results = run_eval(examples, model_fn, scorers)
    summarize(results)

    # 5. Persist for diffing across runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"evals/results/capitals_{timestamp}.jsonl")


if __name__ == "__main__":
    main()
