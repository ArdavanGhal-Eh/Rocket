"""
High-Order Parametric Polynomial Fitting for 3D Missile Trajectories.
=====================================================================
Fits high-order polynomials (degree >= 5) to trajectory coordinates [x(t), y(t), z(t)],
computes statistical determination metrics (R^2), and formats explicit mathematical formulas.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Union
import numpy as np


@dataclass
class ComponentFit:
    """Stores polynomial fit results for a single coordinate component."""
    axis: str
    degree: int
    coefficients: np.ndarray  # Highest power first (numpy standard)
    r2_score: float
    rmse: float
    formula_str: str

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the fitted polynomial at time t."""
        return np.polyval(self.coefficients, t)


@dataclass
class PolynomialFitResult:
    """Full 3D trajectory polynomial fit results."""
    degree: int
    x_fit: ComponentFit
    y_fit: ComponentFit
    z_fit: ComponentFit
    overall_mean_r2: float

    def summary_table(self) -> str:
        """Generate a formatted ASCII presentation of explicit mathematical equations."""
        lines = [
            "=" * 90,
            f"  EXPLICIT PARAMETRIC EQUATIONS OF 3D MISSILE FLIGHT (Order {self.degree} Polynomial)",
            "=" * 90,
            f"Mean Accuracy Metric R^2: {self.overall_mean_r2:.6f}",
            "-" * 90,
            f"[x(t)] R^2 = {self.x_fit.r2_score:.6f} | RMSE = {self.x_fit.rmse:.4f} m",
            f"  x(t) = {self.x_fit.formula_str}",
            "-" * 90,
            f"[y(t)] R^2 = {self.y_fit.r2_score:.6f} | RMSE = {self.y_fit.rmse:.4f} m",
            f"  y(t) = {self.y_fit.formula_str}",
            "-" * 90,
            f"[z(t)] R^2 = {self.z_fit.r2_score:.6f} | RMSE = {self.z_fit.rmse:.4f} m",
            f"  z(t) = {self.z_fit.formula_str}",
            "=" * 90,
        ]
        return "\n".join(lines)


class TrajectoryFitter:
    """
    Performs high-order polynomial regression on 3D flight trajectory data.
    """

    def __init__(self, degree: int = 6):
        if degree < 5:
            raise ValueError(f"Degree must be >= 5 as required by technical specifications. Got {degree}.")
        self.degree = degree

    def _format_formula(self, coeffs: np.ndarray) -> str:
        """Convert polynomial coefficients into a clean, canonical algebraic expression string."""
        deg = len(coeffs) - 1
        terms: List[str] = []
        for i, c in enumerate(coeffs):
            power = deg - i
            if abs(c) < 1e-12:
                continue

            sign_str = "+ " if c >= 0 else "- "
            abs_val = abs(c)

            # Scientific or float formatting based on magnitude
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

    def _fit_single_axis(self, t: np.ndarray, y: np.ndarray, axis_name: str) -> ComponentFit:
        """Fit polynomial for a single axis."""
        coeffs = np.polyfit(t, y, deg=self.degree)
        y_pred = np.polyval(coeffs, t)

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        formula_str = self._format_formula(coeffs)

        return ComponentFit(
            axis=axis_name,
            degree=self.degree,
            coefficients=coeffs,
            r2_score=r2,
            rmse=rmse,
            formula_str=formula_str,
        )

    def fit(self, time: np.ndarray, positions: np.ndarray) -> PolynomialFitResult:
        """
        Fit independent polynomials to x(t), y(t), and z(t).

        Args:
            time: 1D array of time stamps (N,).
            positions: 2D array of positions (N, 3).

        Returns:
            PolynomialFitResult containing individual fits and summary metrics.
        """
        x_fit = self._fit_single_axis(time, positions[:, 0], "x")
        y_fit = self._fit_single_axis(time, positions[:, 1], "y")
        z_fit = self._fit_single_axis(time, positions[:, 2], "z")

        mean_r2 = (x_fit.r2_score + y_fit.r2_score + z_fit.r2_score) / 3.0

        return PolynomialFitResult(
            degree=self.degree,
            x_fit=x_fit,
            y_fit=y_fit,
            z_fit=z_fit,
            overall_mean_r2=mean_r2,
        )
