"""Logging configuration with Rich for nice colored output.

Call `setup_logging()` once at app startup (CLI, notebook, eval runner)
to get pretty tracebacks and structured log output.
"""

import logging

from rich.logging import RichHandler

from quickstart.config import settings


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with Rich handler.

    Idempotent — safe to call multiple times.
    """
    log_level = (level or settings.log_level).upper()

    # Remove any existing handlers to avoid duplicate logs in notebooks
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                show_path=False,
            )
        ],
    )

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
