"""Eval runner — orchestrates dataset → model → scorers → results.

The core loop is intentionally tiny so you can read it end-to-end.
For real workloads, swap the sequential loop for asyncio + a semaphore.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.scorers import Score, Scorer

console = Console()


@dataclass
class Example:
    """A single eval example loaded from a JSONL dataset."""

    id: str
    input: str
    expected: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Result:
    """Result of running a single example."""

    example_id: str
    input: str
    expected: str
    output: str
    scores: dict[str, Score]
    latency_ms: float
    error: str | None = None

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores.values())


def load_dataset(path: str | Path) -> list[Example]:
    """Load a JSONL dataset. One Example per line."""
    examples: list[Example] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        examples.append(
            Example(
                id=data["id"],
                input=data["input"],
                expected=data["expected"],
                tags=data.get("tags", []),
            )
        )
    return examples


def run_eval(
    examples: Iterable[Example],
    model_fn: Callable[[str], str],
    scorers: dict[str, Scorer],
) -> list[Result]:
    """Run examples through model_fn and apply each scorer.

    Parameters
    ----------
    examples:
        Iterable of Example objects.
    model_fn:
        Callable that takes a prompt string and returns the model output.
        This is the unit you're testing. Could wrap a single LLM call,
        an agent, a RAG pipeline — anything string-in, string-out.
    scorers:
        Dict of {name: scorer}. Each scorer runs against every output.
    """
    results: list[Result] = []
    for ex in examples:
        start = time.perf_counter()
        try:
            output = model_fn(ex.input)
            error = None
        except Exception as e:  # noqa: BLE001
            output = ""
            error = f"{type(e).__name__}: {e}"
        latency_ms = (time.perf_counter() - start) * 1000

        scores = {name: scorer(output, ex.expected) for name, scorer in scorers.items()}
        results.append(
            Result(
                example_id=ex.id,
                input=ex.input,
                expected=ex.expected,
                output=output,
                scores=scores,
                latency_ms=latency_ms,
                error=error,
            )
        )
    return results


def summarize(results: list[Result]) -> None:
    """Print a summary table of results to the console."""
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    scorer_names = list(results[0].scores.keys())

    table = Table(title="Eval Results", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Input", style="dim", max_width=40)
    for name in scorer_names:
        table.add_column(name, justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Pass", justify="center")

    for r in results:
        row = [r.example_id, r.input]
        for name in scorer_names:
            row.append(f"{r.scores[name].value:.2f}")
        row.append(f"{r.latency_ms:.0f}ms")
        row.append("[green]✓[/green]" if r.passed else "[red]✗[/red]")
        table.add_row(*row)

    console.print(table)

    # Aggregate stats
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / total
    console.print(
        f"\n[bold]Pass rate:[/bold] {passed}/{total} ({passed / total:.0%})  "
        f"[bold]Avg latency:[/bold] {avg_latency:.0f}ms"
    )

    for name in scorer_names:
        avg_score = sum(r.scores[name].value for r in results) / total
        console.print(f"  [dim]{name}:[/dim] {avg_score:.2f}")


def save_results(results: list[Result], path: str | Path) -> None:
    """Persist results as JSONL for later analysis."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            data = asdict(r)
            # Score is a dataclass — serialize cleanly
            data["scores"] = {k: asdict(v) for k, v in r.scores.items()}
            f.write(json.dumps(data) + "\n")
    console.print(f"[dim]Saved {len(results)} results to {out}[/dim]")
