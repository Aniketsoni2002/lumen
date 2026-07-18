"""Command-line interface for Lumen.

    lumen ingest data/uploads/handbook.pdf
    lumen ask "What is the refund policy, and what is 20% of 30 days?"
    lumen reset
"""
from __future__ import annotations

import argparse
import os
import sys

# The CLI runs single-threaded across separate processes, so persistent SQLite
# memory is both safe and desirable here (it lets --session span multiple runs).
# Set before importing the graph so the checkpointer picks it up. The UI/cloud
# keep the default in-process backend, which is thread-safe.
os.environ.setdefault("LUMEN_MEMORY_BACKEND", "sqlite")

from lumen.agent.graph import run_agent  # noqa: E402
from lumen.core.ingest import ingest_file  # noqa: E402
from lumen.core.vectorstore import clear_collection  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumen", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a document.")
    p_ingest.add_argument("path", help="Path to a PDF/TXT/MD file.")

    p_ask = sub.add_parser("ask", help="Ask the agent a question.")
    p_ask.add_argument("question", help="Your question, in quotes.")
    p_ask.add_argument(
        "--session",
        default=None,
        help="Session id for conversation memory across multiple 'ask' calls.",
    )

    sub.add_parser("reset", help="Clear the knowledge base.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "ingest":
        n = ingest_file(args.path)
        print(f"Indexed {n} chunks from {args.path}")
    elif args.command == "ask":
        result = run_agent(args.question, thread_id=args.session)
        print(result.answer)
        trace = []
        if result.tool_calls:
            trace.append(f"tools used: {', '.join(result.tool_calls)}")
        trace.append(f"steps: {result.steps}")
        if result.reflections:
            trace.append(f"self-corrections: {result.reflections}")
        print(f"\n[{' | '.join(trace)}]")
    elif args.command == "reset":
        clear_collection()
        print("Knowledge base cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
