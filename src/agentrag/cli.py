"""Command-line interface for AgentRAG.

    agentrag ingest data/uploads/handbook.pdf
    agentrag ask "What is the refund policy, and what is 20% of 30 days?"
    agentrag reset
"""
from __future__ import annotations

import argparse
import sys

from agentrag.agent.graph import run_agent
from agentrag.core.ingest import ingest_file
from agentrag.core.vectorstore import clear_collection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentrag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a document.")
    p_ingest.add_argument("path", help="Path to a PDF/TXT/MD file.")

    p_ask = sub.add_parser("ask", help="Ask the agent a question.")
    p_ask.add_argument("question", help="Your question, in quotes.")

    sub.add_parser("reset", help="Clear the knowledge base.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "ingest":
        n = ingest_file(args.path)
        print(f"Indexed {n} chunks from {args.path}")
    elif args.command == "ask":
        result = run_agent(args.question)
        print(result.answer)
        if result.tool_calls:
            print(f"\n[tools used: {', '.join(result.tool_calls)} | "
                  f"steps: {result.steps}]")
    elif args.command == "reset":
        clear_collection()
        print("Knowledge base cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
