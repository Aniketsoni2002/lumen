"""Tests for the safe calculator tool (pure, no models)."""
from __future__ import annotations

import pytest

from lumen.tools.calculator import CalculatorError, calculator, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3", 5),
        ("20 * 3 + 5", 65),
        ("(1000 / 8)", 125.0),
        ("2 ** 10", 1024),
        ("-5 + 2", -3),
        ("17 % 5", 2),
        ("7 // 2", 3),
    ],
)
def test_safe_eval_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "abs(-1)",  # function calls are not allowed
        "x + 1",  # names are not allowed
        "1; 2",  # statements, not an expression
    ],
)
def test_safe_eval_blocks_unsafe_input(malicious):
    with pytest.raises(CalculatorError):
        safe_eval(malicious)


def test_calculator_tool_returns_string_result():
    # .invoke is the LangChain tool interface
    assert calculator.invoke({"expression": "6 * 7"}) == "42"


def test_calculator_tool_reports_errors_gracefully():
    out = calculator.invoke({"expression": "import os"})
    assert out.startswith("CALCULATOR_ERROR")
