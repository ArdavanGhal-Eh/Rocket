"""
3D Missile Dynamics, Propulsion, Aerodynamics, and Mass Depletion.
==================================================================
Models point-mass 3D missile flight dynamics with:
- Variable mass during solid rocket motor burn phase
- Supersonic aerodynamic drag in non-standard atmosphere
- Line-of-sight aligned initial boost
- Spherical lateral acceleration saturation at 35G
"""

from typing import Tuple, Dict, Any
import numpy as np
from .config import MissileConfig


class Missile3D:
    """
    High-fidelity point-mass 3D missile dynamic model.
    """

    def __init__(self, config: MissileConfig):
        self.cfg = config
        self.g_vector = np.array([0.0, 0.0, -self.cfg.g_accel], dtype=np.float64)

    def get_mass(self, t: float) -> float:
        """
        Calculate instantaneous missile mass (kg).

        Linear depletion during burn phase:
            m(t) = m0 - mdot * t    for t <= t_burn
            m(t) = m_dry            for t > t_burn
        """
        if t <= self.cfg.t_burn:
            return float(self.cfg.m0 - self.cfg.mdot * t)
        return float(self.cfg.m_dry)

    def get_thrust(self, t: float, v_unit: np.ndarray) -> np.ndarray:
        """
        Calculate thrust vector aligned with velocity vector.

        Thrust:
            T_vec = T_nominal * v_unit   for t <= t_burn
            T_vec = [0, 0, 0]            for t > t_burn
        """
        if t <= self.cfg.t_burn:
            return self.cfg.thrust_nominal * v_unit
        return np.zeros(3, dtype=np.float64)

    def get_aerodynamic_drag(self, v_mag: float, v_unit: np.ndarray) -> np.ndarray:
        """
        Calculate 3D aerodynamic drag vector opposing velocity.

        Formula:
            F_D = -0.5 * rho * |v|^2 * C_D * A * v_hat
        """
        # Dynamic pressure q = 0.5 * rho * v^2
        dyn_pressure = 0.5 * self.cfg.rho * (v_mag ** 2)
        drag_force = dyn_pressure * self.cfg.cd_default * self.cfg.ref_area
        return -drag_force * v_unit

    def enforce_spherical_g_limit(
        self,
        a_cmd: np.ndarray,
        v_unit: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Apply spherical saturation to commanded lateral acceleration (max 35G).

        Guidance produces lateral (normal) acceleration perpendicular to velocity:
            a_lateral = a_cmd - (a_cmd . v_hat) * v_hat
        Spherical limiter:
            if |a_lateral| > a_max:
                a_lateral = a_max * (a_lateral / |a_lateral|)

        Returns:
            Tuple of (saturated_a_lateral, is_saturated)
        """
        # Project command to normal plane of velocity vector
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
        """
        Compute all force and acceleration components acting on the missile.

        Returns:
            Dictionary with breakdown of thrust, drag, guidance, gravity, and net acceleration.
        """
        v_mag = float(np.linalg.norm(v_M))
        if v_mag < 1e-4:
            v_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            v_unit = v_M / v_mag

        m = self.get_mass(t)
        thrust_vec = self.get_thrust(t, v_unit)
        drag_vec = self.get_aerodynamic_drag(v_mag, v_unit)

        # Guidance lateral acceleration with spherical 35G limit
        a_lateral, is_saturated = self.enforce_spherical_g_limit(a_cmd, v_unit)

        # Net acceleration: a_net = (Thrust + Drag)/m + a_lateral + g
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
