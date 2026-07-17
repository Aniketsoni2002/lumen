"""A safe calculator tool.

LLMs are unreliable at arithmetic, so we give the agent a real evaluator. It is
locked down to a small math-only grammar via the ``ast`` module — no ``eval``,
no access to names, calls, or attributes — so a malicious or confused query
cannot execute arbitrary code.
"""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

from agentrag.utils.logging import get_logger

logger = get_logger("tool.calculator")

# Only these operators are permitted.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised for an expression that isn't safe, valid arithmetic."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression safely. Raises CalculatorError."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {expression!r}") from exc
    return _eval_node(tree)


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '20 * 3 + 5' or '(1000 / 8) ** 2'.

    Use this whenever a question requires exact arithmetic. Supports + - * / //
    % and ** with parentheses. Do NOT do the math yourself — use this tool.
    """
    try:
        result = safe_eval(expression)
    except CalculatorError as exc:
        return f"CALCULATOR_ERROR: {exc}"
    logger.info("Calculated %s = %s", expression, result)
    return str(result)
