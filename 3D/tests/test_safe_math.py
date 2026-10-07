"""
Unit tests for SafeMathExpression (3D GNC Suite).
Verifies parsing validity, numerical fidelity, and strict security sandboxing.
"""

import os
import sys
import math
import unittest
import numpy as np

# Ensure 3D directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from missile_sim.safe_math import SafeMathExpression, SecurityError


class TestSafeMathExpression(unittest.TestCase):
    """Test suite for safe AST-based mathematical formula evaluation."""

    def test_basic_arithmetic_and_constants(self):
        expr = SafeMathExpression("100 + 50 * t - 25 * t^2")
        self.assertAlmostEqual(expr.evaluate(0.0), 100.0)
        self.assertAlmostEqual(expr.evaluate(1.0), 125.0)
        self.assertAlmostEqual(expr.evaluate(2.0), 100.0)

    def test_trigonometric_and_transcendental_functions(self):
        expr = SafeMathExpression("sin(0.5 * pi * t) + exp(0.1 * t) + sqrt(16 * t)")
        # At t = 1.0: sin(pi/2) + exp(0.1) + sqrt(16) = 1 + exp(0.1) + 4
        expected = 1.0 + math.exp(0.1) + 4.0
        self.assertAlmostEqual(expr.evaluate(1.0), expected, places=5)

    def test_array_evaluation(self):
        expr = SafeMathExpression("2000 - 150 * t")
        t_arr = np.array([0.0, 1.0, 2.0, 3.0])
        res = expr.evaluate(t_arr)
        np.testing.assert_allclose(res, np.array([2000.0, 1850.0, 1700.0, 1550.0]))

    def test_security_attribute_access_blocked(self):
        """Verify that attribute access (e.g. __class__) is strictly forbidden."""
        malicious_inputs = [
            "(1).__class__",
            "t.__class__.__bases__",
            "t.real",
            "sin.__code__",
        ]
        for bad_expr in malicious_inputs:
            with self.assertRaises((SecurityError, ValueError)):
                SafeMathExpression(bad_expr)

    def test_security_arbitrary_function_calls_blocked(self):
        """Verify that arbitrary builtins or dangerous function calls are blocked."""
        disallowed_inputs = [
            "__import__('os').system('dir')",
            "open('README.md')",
            "eval('1+1')",
            "exec('x=1')",
            "exit()",
            "print(t)",
        ]
        for bad_expr in disallowed_inputs:
            with self.assertRaises((SecurityError, ValueError)):
                SafeMathExpression(bad_expr)

    def test_security_unknown_identifiers_blocked(self):
        """Verify that undeclared variables/identifiers are blocked."""
        with self.assertRaises((SecurityError, ValueError)):
            SafeMathExpression("unknown_var * t")

    def test_syntax_error_handling(self):
        """Verify that malformed expressions raise informative ValueError."""
        with self.assertRaises(ValueError):
            SafeMathExpression("sin(t + * 5)")


if __name__ == "__main__":
    unittest.main()
