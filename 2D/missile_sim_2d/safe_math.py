"""
Safe Mathematical Expression Parser and Pre-compiled Evaluator for 2D Simulations.
==================================================================================
Provides high-performance, sandboxed evaluation of analytical mathematical expressions
for custom target trajectory modeling without relying on unsafe arbitrary `eval()`.
"""

import ast
from typing import Dict, Any, Callable, Union, Set
import numpy as np

# Whitelist of allowed mathematical function names
ALLOWED_MATH_FUNCS: Dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
}

# Whitelist of allowed constant symbols
ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": float(np.pi),
    "e": float(np.e),
}

# Whitelist of allowed AST node types
ALLOWED_AST_NODES: Set[type] = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    # Binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # Unary operators
    ast.UAdd,
    ast.USub,
    # Contexts
    ast.Load,
}


class SafeMathExpression2D:
    """
    Parses, validates, and pre-compiles a mathematical expression string f(t) in 2D.
    Guarantees zero arbitrary code execution risk while optimizing evaluation speed.
    """

    def __init__(self, expression_str: str, variable_name: str = "t"):
        self.raw_expression = expression_str.strip()
        self.variable_name = variable_name

        # Standardize power operator (^ -> **)
        self.clean_expression = self.raw_expression.replace("^", "**")

        # Parse AST and validate security
        self.tree = self._parse_and_validate(self.clean_expression)

        # Pre-compile into bytecode for rapid evaluation inside RK4 loops
        self._compiled_code = compile(self.tree, filename="<safe_math_2d>", mode="eval")

        # Prepare execution namespace
        self._namespace: Dict[str, Any] = {
            "__builtins__": {},
            **ALLOWED_MATH_FUNCS,
            **ALLOWED_CONSTANTS,
        }

    def _parse_and_validate(self, expr_str: str) -> ast.Expression:
        """Parse expression into AST and strictly enforce whitelist validation."""
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"خطای نحوی در فرمول ریاضی '{self.raw_expression}': {e}") from e

        self._validate_node(tree)
        return tree

    def _validate_node(self, node: ast.AST) -> None:
        """Recursively validate every AST node against the security whitelist."""
        if type(node) not in ALLOWED_AST_NODES:
            raise SecurityError(
                f"کاراکتر یا گره دستوری غیرمجاز در فرمول ریاضی: {type(node).__name__}. "
                f"فقط اعمال ریاضی و توابع مجاز پشتیبانی می‌شوند."
            )

        if isinstance(node, ast.Expression):
            self._validate_node(node.body)

        elif isinstance(node, ast.BinOp):
            self._validate_node(node.left)
            self._validate_node(node.op)
            self._validate_node(node.right)

        elif isinstance(node, ast.UnaryOp):
            self._validate_node(node.op)
            self._validate_node(node.operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SecurityError("فراخوانی توابع فقط از طریق نام مجاز است (دسترسی به ویژگی‌ها مسدود است).")
            func_name = node.func.id
            if func_name not in ALLOWED_MATH_FUNCS:
                raise SecurityError(
                    f"تابع '{func_name}' در لیست توابع مجاز ریاضی قرار ندارد. "
                    f"توابع مجاز: {sorted(list(ALLOWED_MATH_FUNCS.keys()))}"
                )
            for arg in node.args:
                self._validate_node(arg)
            if node.keywords:
                raise SecurityError("آرگومان‌های نام‌دار در توابع ریاضی مجاز نیستند.")

        elif isinstance(node, ast.Name):
            allowed_names = set(ALLOWED_MATH_FUNCS.keys()) | set(ALLOWED_CONSTANTS.keys()) | {self.variable_name}
            if node.id not in allowed_names:
                raise SecurityError(
                    f"متغیر یا شناسه ناشناخته '{node.id}' در فرمول ریاضی. "
                    f"تنها متغیر مجاز '{self.variable_name}' و ثوابت ریاضی هستند."
                )

        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise SecurityError(f"نوع مقدار ثابت {type(node.value).__name__} مجاز نیست. فقط اعداد مجازند.")

    def evaluate(self, t: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate pre-compiled expression at value t.

        Args:
            t: Scalar float or NumPy array of time values.

        Returns:
            Computed numerical value as float or np.ndarray.
        """
        self._namespace[self.variable_name] = t
        try:
            return eval(self._compiled_code, {"__builtins__": {}}, self._namespace)
        except Exception as e:
            raise ValueError(
                f"خطا هنگام محاسبه عددی فرمول '{self.raw_expression}' در t={t}: {e}"
            ) from e

    def __call__(self, t: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.evaluate(t)


class SecurityError(ValueError):
    """Raised when a mathematical formula contains unauthorized or potentially hazardous syntax."""
    pass
