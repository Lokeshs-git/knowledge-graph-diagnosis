"""Command-line interface for the quickstart project.

Run with: `uv run qs --help`

Add new commands by writing functions and decorating them with @app.command().
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.tree import Tree

from quickstart.llm import LLMClient
from quickstart.logging import setup_logging

app = typer.Typer(
    help="Quickstart CLI — replace this with your project's commands.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Prompt to send to the model."),
    model: str | None = typer.Option(None, "--model", "-m", help="Override default model."),
) -> None:
    """Send a one-shot prompt to the LLM and print the reply."""
    setup_logging()
    client = LLMClient(model=model)
    reply = client.complete(prompt)
    console.print(reply)


@app.command()
def version() -> None:
    """Print the package version."""
    from quickstart import __version__

    console.print(f"quickstart {__version__}")


@app.command()
def trace(
    artifact_id: str = typer.Argument(..., help="Artifact ID to trace, e.g. PRD-003 or TOPIC-overview."),
    depth: int = typer.Option(3, "--depth", "-d", help="Maximum traversal depth."),
) -> None:
    """Walk the knowledge graph from an artifact and print connected nodes.

    Shows parents, children, related topics, and code references for the
    given ID. Useful for understanding what depends on what.

    Implements: ADR-0002 (knowledge graph in markdown)
    """
    # Import here to keep CLI startup fast for non-trace commands
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    try:
        from check_links import (  # type: ignore[import-not-found]
            _as_id_list,
            find_code_refs,
            find_markdown_artifacts,
        )
    except ImportError:
        console.print("[red]Error:[/red] tools/check_links.py not found")
        raise typer.Exit(1) from None

    root = Path(__file__).resolve().parents[2]
    artifacts = find_markdown_artifacts(root)
    code_refs = find_code_refs(root)

    if artifact_id not in artifacts:
        available = sorted(artifacts.keys())
        console.print(f"[red]Unknown artifact:[/red] {artifact_id}")
        console.print(f"\nAvailable IDs: {', '.join(available)}")
        raise typer.Exit(1)

    root_artifact = artifacts[artifact_id]
    title = _read_title(root_artifact.body) or root_artifact.type
    tree = Tree(f"[bold]{artifact_id}[/bold] — {title} [dim]({root_artifact.path})[/dim]")

    visited: set[str] = {artifact_id}
    _expand(tree, root_artifact, artifacts, code_refs, visited, depth, _as_id_list)
    console.print(tree)


def _expand(
    parent_tree: Tree,
    artifact,  # noqa: ANN001 — Artifact type imported lazily
    artifacts: dict,
    code_refs: dict,
    visited: set[str],
    depth: int,
    as_id_list,  # noqa: ANN001
) -> None:
    """Recursively expand graph nodes into the rich Tree."""
    if depth <= 0:
        return

    # Parents
    parents = as_id_list(artifact.frontmatter.get("parents"))
    if parents:
        parent_branch = parent_tree.add("[yellow]parents[/yellow]")
        for pid in parents:
            if pid in visited:
                parent_branch.add(f"[dim]{pid} (already shown)[/dim]")
                continue
            visited.add(pid)
            if pid not in artifacts:
                parent_branch.add(f"[red]{pid} (missing)[/red]")
                continue
            child_artifact = artifacts[pid]
            child_title = _read_title(child_artifact.body) or child_artifact.type
            sub = parent_branch.add(f"[bold]{pid}[/bold] — {child_title}")
            _expand(sub, child_artifact, artifacts, code_refs, visited, depth - 1, as_id_list)

    # Children
    children = as_id_list(artifact.frontmatter.get("children"))
    if children:
        child_branch = parent_tree.add("[green]children[/green]")
        for cid in children:
            if cid in visited:
                child_branch.add(f"[dim]{cid} (already shown)[/dim]")
                continue
            visited.add(cid)
            if cid not in artifacts:
                child_branch.add(f"[red]{cid} (missing)[/red]")
                continue
            ca = artifacts[cid]
            ca_title = _read_title(ca.body) or ca.type
            sub = child_branch.add(f"[bold]{cid}[/bold] — {ca_title}")
            _expand(sub, ca, artifacts, code_refs, visited, depth - 1, as_id_list)

    # Related topics
    related = as_id_list(artifact.frontmatter.get("related-topics"))
    if related:
        related_branch = parent_tree.add("[cyan]related[/cyan]")
        for rid in related:
            if rid not in artifacts:
                related_branch.add(f"[red]{rid} (missing)[/red]")
                continue
            ra = artifacts[rid]
            ra_title = _read_title(ra.body) or ra.type
            related_branch.add(f"{rid} — {ra_title}")

    # Code references
    if artifact.id in code_refs:
        code_branch = parent_tree.add("[magenta]code references[/magenta]")
        for path, lineno, _ in code_refs[artifact.id]:
            code_branch.add(f"{path}:{lineno}")


def _read_title(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


if __name__ == "__main__":
    app()
