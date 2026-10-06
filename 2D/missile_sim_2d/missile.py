"""
2D Missile Dynamics, Propulsion, Aerodynamics, and Mass Depletion.
==================================================================
Models point-mass 2D missile flight dynamics with:
- Variable mass during solid rocket motor burn phase
- Supersonic aerodynamic drag
- Circular lateral acceleration saturation at 35G
"""

from typing import Tuple, Dict, Any
import numpy as np
from .config import MissileConfig2D


class Missile2D:
    """High-fidelity point-mass 2D missile dynamic model."""

    def __init__(self, config: MissileConfig2D):
        self.cfg = config
        # Standard gravity in 2D (zero for symmetric planar engagement, or active if requested)
        if getattr(self.cfg, "include_gravity", False):
            self.g_vector = np.array([0.0, -self.cfg.g_accel], dtype=np.float64)
        else:
            self.g_vector = np.zeros(2, dtype=np.float64)

    def get_mass(self, t: float) -> float:
        """Calculate instantaneous missile mass (kg)."""
        if t <= self.cfg.t_burn:
            return float(self.cfg.m0 - self.cfg.mdot * t)
        return float(self.cfg.m_dry)

    def get_thrust(self, t: float, v_unit: np.ndarray) -> np.ndarray:
        """Calculate thrust vector aligned with velocity vector."""
        if t <= self.cfg.t_burn:
            return self.cfg.thrust_nominal * v_unit
        return np.zeros(2, dtype=np.float64)

    def get_aerodynamic_drag(self, v_mag: float, v_unit: np.ndarray) -> np.ndarray:
        """Calculate 2D aerodynamic drag vector opposing velocity."""
        dyn_pressure = 0.5 * self.cfg.rho * (v_mag ** 2)
        drag_force = dyn_pressure * self.cfg.cd_default * self.cfg.ref_area
        return -drag_force * v_unit

    def enforce_g_limit(
        self,
        a_cmd: np.ndarray,
        v_unit: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Apply saturation to commanded lateral acceleration (max 35G).
        Guidance produces lateral acceleration perpendicular to velocity:
            a_lateral = a_cmd - (a_cmd . v_hat) * v_hat
        """
        a_lateral = a_cmd - np.dot(a_cmd, v_unit) * v_unit
        lat_mag = float(np.linalg.norm(a_lateral))

        max_allowed = self.cfg.max_lateral_accel
        if lat_mag > max_allowed:
            a_lateral_saturated = max_allowed * (a_lateral / lat_mag)
            return a_lateral_saturated, True

        return a_lateral, False

    def compute_accelerations(
        self,
        t: float,
        v_M: np.ndarray,
        a_cmd: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all acceleration components in 2D."""
        v_mag = float(np.linalg.norm(v_M))
        if v_mag < 1e-4:
            v_unit = np.array([1.0, 0.0], dtype=np.float64)
        else:
            v_unit = v_M / v_mag

        m = self.get_mass(t)
        thrust_vec = self.get_thrust(t, v_unit)
        drag_vec = self.get_aerodynamic_drag(v_mag, v_unit)

        a_lateral, is_saturated = self.enforce_g_limit(a_cmd, v_unit)

        a_propulsion_aero = (thrust_vec + drag_vec) / m
        a_net = a_propulsion_aero + a_lateral + self.g_vector

        return {
            "mass": m,
            "v_mag": v_mag,
            "v_unit": v_unit,
            "thrust_vec": thrust_vec,
            "drag_vec": drag_vec,
            "a_lateral": a_lateral,
            "a_lateral_mag": float(np.linalg.norm(a_lateral)),
            "is_saturated": is_saturated,
            "a_net": a_net,
            "a_net_mag": float(np.linalg.norm(a_net)),
        }
