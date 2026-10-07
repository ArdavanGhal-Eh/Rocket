"""
Unit tests for SafeMathExpression2D (2D GNC Suite).
Verifies parsing validity, numerical fidelity, and strict security sandboxing.
"""

import os
import sys
import math
import unittest
import numpy as np

# Ensure 2D directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from missile_sim_2d.safe_math import SafeMathExpression2D, SecurityError


class TestSafeMathExpression2D(unittest.TestCase):
    """Test suite for safe AST-based mathematical formula evaluation in 2D."""

    def test_basic_arithmetic_and_constants(self):
        expr = SafeMathExpression2D("6000 - 250*t + 50*t^2")
        self.assertAlmostEqual(expr.evaluate(0.0), 6000.0)
        self.assertAlmostEqual(expr.evaluate(1.0), 5800.0)
        self.assertAlmostEqual(expr.evaluate(2.0), 5700.0)

    def test_trigonometric_functions(self):
        expr = SafeMathExpression2D("2500 + 400 * sin(0.5 * t)")
        self.assertAlmostEqual(expr.evaluate(0.0), 2500.0)
        self.assertAlmostEqual(expr.evaluate(math.pi), 2500.0 + 400.0 * math.sin(0.5 * math.pi))

    def test_security_blocked_exploits(self):
        exploits = [
            "(1).__class__",
            "__import__('os').system('ls')",
            "open('test.txt')",
            "eval('2+2')",
            "exec('pass')",
            "bad_variable + t",
        ]
        for exp in exploits:
            with self.assertRaises((SecurityError, ValueError)):
                SafeMathExpression2D(exp)


if __name__ == "__main__":
    unittest.main()
