"""
Parametric Polynomial Fitting for 2D Missile Trajectories.
==========================================================
Fits high-order polynomials (degree >= 5) to coordinates [x(t), y(t)],
computes statistical determination metrics (R^2), and formats explicit mathematical formulas.
"""

from dataclasses import dataclass
from typing import List, Union
import numpy as np


@dataclass
class ComponentFit2D:
    """Stores polynomial fit results for a single coordinate component in 2D."""
    axis: str
    degree: int
    coefficients: np.ndarray
    r2_score: float
    rmse: float
    formula_str: str

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.polyval(self.coefficients, t)


@dataclass
class PolynomialFitResult2D:
    """Full 2D trajectory polynomial fit results."""
    degree: int
    x_fit: ComponentFit2D
    y_fit: ComponentFit2D
    overall_mean_r2: float

    def summary_table(self) -> str:
        lines = [
            "=" * 90,
            f"  EXPLICIT PARAMETRIC EQUATIONS OF 2D MISSILE FLIGHT (Order {self.degree} Polynomial)",
            "=" * 90,
            f"Mean Accuracy Metric R^2: {self.overall_mean_r2:.6f}",
            "-" * 90,
            f"[x(t)] R^2 = {self.x_fit.r2_score:.6f} | RMSE = {self.x_fit.rmse:.4f} m",
            f"  x(t) = {self.x_fit.formula_str}",
            "-" * 90,
            f"[y(t)] R^2 = {self.y_fit.r2_score:.6f} | RMSE = {self.y_fit.rmse:.4f} m",
            f"  y(t) = {self.y_fit.formula_str}",
            "=" * 90,
        ]
        return "\n".join(lines)


class TrajectoryFitter2D:
    """Performs high-order polynomial regression on 2D flight trajectory data."""

    def __init__(self, degree: int = 6):
        if degree < 5:
            raise ValueError(f"Degree must be >= 5. Got {degree}.")
        self.degree = degree

    def _format_formula(self, coeffs: np.ndarray) -> str:
        deg = len(coeffs) - 1
        terms: List[str] = []
        for i, c in enumerate(coeffs):
            power = deg - i
            if abs(c) < 1e-12:
                continue

            sign_str = "+ " if c >= 0 else "- "
            abs_val = abs(c)

            if abs_val >= 1e4 or (abs_val < 1e-2 and abs_val > 0):
                c_str = f"{abs_val:.4e}"
            else:
                c_str = f"{abs_val:.4f}"

            if power > 1:
                term = f"{sign_str}{c_str}*t^{power}"
            elif power == 1:
                term = f"{sign_str}{c_str}*t"
            else:
                term = f"{sign_str}{c_str}"

            terms.append(term)

        if not terms:
            return "0.0"

        formula = " ".join(terms)
        if formula.startswith("+ "):
            formula = formula[2:]
        return formula

    def _fit_single_axis(self, t: np.ndarray, y: np.ndarray, axis_name: str) -> ComponentFit2D:
        coeffs = np.polyfit(t, y, deg=self.degree)
        y_pred = np.polyval(coeffs, t)

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        formula_str = self._format_formula(coeffs)

        return ComponentFit2D(
            axis=axis_name,
            degree=self.degree,
            coefficients=coeffs,
            r2_score=r2,
            rmse=rmse,
            formula_str=formula_str,
        )

    def fit(self, time: np.ndarray, positions: np.ndarray) -> PolynomialFitResult2D:
        x_fit = self._fit_single_axis(time, positions[:, 0], "x")
        y_fit = self._fit_single_axis(time, positions[:, 1], "y")
        mean_r2 = (x_fit.r2_score + y_fit.r2_score) / 2.0

        return PolynomialFitResult2D(
            degree=self.degree,
            x_fit=x_fit,
            y_fit=y_fit,
            overall_mean_r2=mean_r2,
        )
