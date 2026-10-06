"""
Modular 3D Target Maneuver Trajectories for Fighter Aircraft.
=============================================================
Provides distinct physical, parametric, and modular 3D target trajectories:
1. Scenario A: StraightLineTarget (Uniform non-maneuvering flight)
2. Scenario B: BarrelRollSpiralTarget (3D barrel roll and spiral climb)
3. Scenario C: HighGInclinedTurnTarget (High-G sustained turn in an inclined plane)
4. Scenario D: HighG3DSTurnTarget (High-G defensive S-turn/Break-turns with smoothed square wave)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type
import numpy as np
from .config import TargetConfig


class BaseTargetManeuver(ABC):
    """Abstract base class for modular 3D target trajectories."""

    def __init__(self, config: TargetConfig):
        self.config = config
        self.r0 = config.r0.copy()
        self.speed = config.speed
        self.name = "Base Maneuver"
        self.description = "Base parametric trajectory"

    @abstractmethod
    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute target kinematics at time t.

        Args:
            t: Elapsed simulation time (s).

        Returns:
            Tuple of:
                r_T: Position vector [x, y, z] (m)
                v_T: Velocity vector [vx, vy, vz] (m/s)
                a_T: Acceleration vector [ax, ay, az] (m/s^2)
        """
        pass

    def __call__(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_state(t)


class StraightLineTarget(BaseTargetManeuver):
    """
    Scenario A: Uniform straight line flight along a constant 3D heading vector.
    Target cruises at constant speed with zero acceleration.
    """

    def __init__(self, config: TargetConfig):
        super().__init__(config)
        self.name = "Scenario A: Straight Line Flight"
        self.description = "Uniform, non-accelerating flight along a constant 3D heading vector."
        self.v_dir = self.config.initial_heading / np.linalg.norm(self.config.initial_heading)
        self.v_const = self.speed * self.v_dir

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_T = self.r0 + self.v_const * t
        v_T = self.v_const.copy()
        a_T = np.zeros(3, dtype=np.float64)
        return r_T, v_T, a_T


class BarrelRollSpiralTarget(BaseTargetManeuver):
    """
    Scenario B: 3D Barrel Roll and Spiral Climb.
    The aircraft advances along an inclined flight axis while executing a continuous
    barrel roll around that axis at a constant roll rate and high sustained normal load.
    """

    def __init__(self, config: TargetConfig, roll_radius: float = 220.0, roll_rate: float = 0.58):
        super().__init__(config)
        self.name = "Scenario B: 3D Barrel Roll / Spiral Climb"
        self.description = (
            "Continuous helical roll around an inclined climbing axis with ~7.6G normal acceleration."
        )
        self.roll_radius = roll_radius
        self.omega = roll_rate

        # Tangential speed around helix: v_theta = omega * roll_radius
        v_theta = self.omega * self.roll_radius
        # Axial speed to maintain total aircraft speed of exactly 300 m/s:
        if v_theta >= self.speed:
            raise ValueError("Tangential roll speed exceeds total aircraft speed.")
        self.v_axial = np.sqrt(self.speed ** 2 - v_theta ** 2)

        # Inclined climb axis (e.g. heading toward missile with climb angle)
        climb_dir = np.array([-0.75, -0.4, 0.45], dtype=np.float64)
        self.e_axis = climb_dir / np.linalg.norm(climb_dir)

        # Build orthonormal basis (e_axis, u1, u2)
        ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u1 = np.cross(self.e_axis, ref_up)
        if np.linalg.norm(u1) < 1e-4:
            ref_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u1 = np.cross(self.e_axis, ref_up)
        self.u1 = u1 / np.linalg.norm(u1)
        self.u2 = np.cross(self.e_axis, self.u1)
        self.u2 /= np.linalg.norm(self.u2)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.omega * t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Position: axial advance + circular roll offset (anchored at t=0)
        r_T = (
            self.r0
            + (self.v_axial * t) * self.e_axis
            + self.roll_radius * ((cos_t - 1.0) * self.u1 + sin_t * self.u2)
        )

        # Velocity
        v_T = (
            self.v_axial * self.e_axis
            + (self.roll_radius * self.omega) * (-sin_t * self.u1 + cos_t * self.u2)
        )

        # Acceleration (strictly centripetal normal acceleration)
        a_T = -(self.roll_radius * (self.omega ** 2)) * (cos_t * self.u1 + sin_t * self.u2)

        return r_T, v_T, a_T


class HighGInclinedTurnTarget(BaseTargetManeuver):
    """
    Scenario C: High-G Sustained Turn in an Inclined Plane.
    The fighter executes a maximum performance circular turn banked at an incline
    pulling ~7.5G lateral acceleration.
    """

    def __init__(self, config: TargetConfig, g_pull: float = 7.5, bank_angle_deg: float = 45.0):
        super().__init__(config)
        self.name = "Scenario C: High-G Inclined Turn"
        self.description = f"Sustained {g_pull:.1f}G circular turn in an inclined maneuver plane."
        self.a_n = g_pull * 9.81
        self.turn_radius = (self.speed ** 2) / self.a_n
        self.omega = self.speed / self.turn_radius

        # Inclined plane definition: normal vector tilted by bank angle
        bank_rad = np.radians(bank_angle_deg)
        # Plane normal tilted relative to vertical
        self.plane_normal = np.array([0.0, np.sin(bank_rad), np.cos(bank_rad)], dtype=np.float64)
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        # Heading at t=0
        v0_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        # Ensure v0_dir is perpendicular to plane_normal
        v0_dir = v0_dir - np.dot(v0_dir, self.plane_normal) * self.plane_normal
        self.u_tan0 = v0_dir / np.linalg.norm(v0_dir)

        # Radial unit vector at t=0 directed inward to center of turn
        self.u_rad0 = np.cross(self.plane_normal, self.u_tan0)
        self.u_rad0 /= np.linalg.norm(self.u_rad0)

        # Turn center
        self.center = self.r0 + self.turn_radius * self.u_rad0

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.omega * t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Position on circle
        r_T = self.center - self.turn_radius * (cos_t * self.u_rad0 - sin_t * self.u_tan0)

        # Velocity
        v_T = self.speed * (sin_t * self.u_rad0 + cos_t * self.u_tan0)

        # Centripetal acceleration directed toward center
        a_T = self.a_n * (cos_t * self.u_rad0 - sin_t * self.u_tan0)

        return r_T, v_T, a_T


class HighG3DSTurnTarget(BaseTargetManeuver):
    """
    Scenario D: High-G 3D S-Turn / Break Turns.
    Rapid alternating evasion maneuvers with 7G to 9G normal acceleration
    using smoothed square wave profiles across both horizontal and vertical axes,
    with a period of 1.8 seconds.
    """

    def __init__(
        self,
        config: TargetConfig,
        period: float = 1.8,
        g_horizontal: float = 8.5,
        g_vertical: float = 7.5,
        t_precompute: float = 30.0,
        dt_precompute: float = 0.001,
    ):
        super().__init__(config)
        self.name = "Scenario D: High-G 3D S-Turn / Break Turns"
        self.description = (
            f"Violent 3D defensive weave ({g_horizontal:.1f}G lateral, {g_vertical:.1f}G vertical) "
            f"with period {period:.1f}s smoothed square waves."
        )
        self.period = period
        self.omega_s = 2.0 * np.pi / period
        self.a_h_mag = g_horizontal * 9.81
        self.a_v_mag = g_vertical * 9.81

        # Base flight direction
        base_dir = self.config.initial_heading / np.linalg.norm(self.config.initial_heading)
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.lat_ref = np.cross(base_dir, up_ref)
        self.lat_ref /= np.linalg.norm(self.lat_ref)
        self.vert_ref = np.cross(self.lat_ref, base_dir)
        self.vert_ref /= np.linalg.norm(self.vert_ref)

        # High-fidelity precomputation of trajectory via RK4 to maintain exact speed 300 m/s
        self.dt_prec = dt_precompute
        steps = int(t_precompute / dt_precompute) + 1
        self.times = np.linspace(0, t_precompute, steps)
        self.r_table = np.zeros((steps, 3), dtype=np.float64)
        self.v_table = np.zeros((steps, 3), dtype=np.float64)
        self.a_table = np.zeros((steps, 3), dtype=np.float64)

        curr_r = self.r0.copy()
        curr_v = self.speed * base_dir.copy()

        tanh_scale = np.tanh(3.0)

        for i, t_val in enumerate(self.times):
            # Commanded normal accelerations via smoothed square wave
            sq_h = np.tanh(3.0 * np.sin(self.omega_s * t_val)) / tanh_scale
            sq_v = np.tanh(3.0 * np.cos(self.omega_s * t_val)) / tanh_scale

            raw_a = (self.a_h_mag * sq_h) * self.lat_ref + (self.a_v_mag * sq_v) * self.vert_ref

            # Ensure pure normal acceleration (orthogonal to current velocity)
            v_norm = np.linalg.norm(curr_v)
            v_unit = curr_v / v_norm
            a_normal = raw_a - np.dot(raw_a, v_unit) * v_unit

            self.r_table[i] = curr_r
            self.v_table[i] = curr_v
            self.a_table[i] = a_normal

            # Integration step
            curr_r += curr_v * dt_precompute
            curr_v += a_normal * dt_precompute
            # Maintain strict speed invariance
            curr_v = self.speed * (curr_v / np.linalg.norm(curr_v))

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate precomputed continuous trajectory."""
        if t <= 0.0:
            return self.r_table[0].copy(), self.v_table[0].copy(), self.a_table[0].copy()
        if t >= self.times[-1]:
            return self.r_table[-1].copy(), self.v_table[-1].copy(), self.a_table[-1].copy()

        idx = int(t / self.dt_prec)
        idx = max(0, min(idx, len(self.times) - 2))
        frac = (t - self.times[idx]) / self.dt_prec

        r_T = self.r_table[idx] * (1.0 - frac) + self.r_table[idx + 1] * frac
        v_T = self.v_table[idx] * (1.0 - frac) + self.v_table[idx + 1] * frac
        a_T = self.a_table[idx] * (1.0 - frac) + self.a_table[idx + 1] * frac

        return r_T, v_T, a_T


class LinearEquationTarget3D(BaseTargetManeuver):
    """
    معادله خط سه‌بعدی هدف (3D Linear Equation Target Trajectory):
    مسیر خطی در فضای سه‌بعدی اقلیدسی به فرم برداری:
        r_T(t) = r0 + v_const * t
    با سرعت ثابت speed و بردار جهت heading_dir.
    """

    def __init__(
        self,
        config: TargetConfig,
        r0: np.ndarray = None,
        heading_dir: np.ndarray = None,
        speed: float = 300.0,
        custom_name: str = "3D Linear Path",
    ):
        super().__init__(config)
        if r0 is not None:
            self.r0 = np.array(r0, dtype=np.float64)
        else:
            self.r0 = self.config.r0.copy()

        self.speed = float(speed)
        if heading_dir is not None:
            h = np.array(heading_dir, dtype=np.float64)
        else:
            h = self.config.initial_heading.copy()

        h_norm = np.linalg.norm(h)
        self.v_dir = h / (h_norm if h_norm > 1e-6 else 1.0)
        self.v_const = self.speed * self.v_dir
        self.a_const = np.zeros(3, dtype=np.float64)

        self.equation_str = (
            f"r(t) = [{self.r0[0]:.0f}, {self.r0[1]:.0f}, {self.r0[2]:.0f}] + "
            f"[{self.v_const[0]:.1f}, {self.v_const[1]:.1f}, {self.v_const[2]:.1f}]·t"
        )
        self.name = f"{custom_name}: {self.equation_str}"
        self.description = (
            f"پرواز هدف بر روی خط مستقیم سه‌بعدی با سرعت {self.speed:.1f} m/s و جهت {self.v_dir}."
        )

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_T = self.r0 + self.v_const * t
        v_T = self.v_const.copy()
        a_T = self.a_const.copy()
        return r_T, v_T, a_T


class ParabolicEquationTarget3D(BaseTargetManeuver):
    """
    معادله سهمی سه‌بعدی هدف (3D Parabolic Equation Target Trajectory):
    مسیر پرواز سهموی در فضای سه‌بعدی تحت شتاب برداری ثابت (مانند جاذبه یا شتاب مانور زاویه‌دار):
        r_T(t) = r0 + v0 * t + 0.5 * a_const * t^2
        v_T(t) = v0 + a_const * t
        a_T(t) = a_const
    """

    def __init__(
        self,
        config: TargetConfig,
        r0: np.ndarray = None,
        v0: np.ndarray = None,
        a_const: np.ndarray = None,
        custom_name: str = "3D Parabolic Path",
    ):
        super().__init__(config)
        self.r0 = np.array(r0, dtype=np.float64) if r0 is not None else self.config.r0.copy()
        if v0 is not None:
            self.v0 = np.array(v0, dtype=np.float64)
        else:
            self.v0 = np.array([-240.0, -120.0, 60.0], dtype=np.float64)

        if a_const is not None:
            self.a_const = np.array(a_const, dtype=np.float64)
        else:
            self.a_const = np.array([0.0, 15.0, -9.81], dtype=np.float64)

        self.equation_str = (
            f"r(t) = r0 + [{self.v0[0]:.1f}, {self.v0[1]:.1f}, {self.v0[2]:.1f}]·t + "
            f"0.5·[{self.a_const[0]:.1f}, {self.a_const[1]:.1f}, {self.a_const[2]:.1f}]·t²"
        )
        self.name = f"{custom_name}: {self.equation_str}"
        self.description = (
            f"Parabolic 3D target flight with constant acceleration vector [{self.a_const[0]:.1f}, {self.a_const[1]:.1f}, {self.a_const[2]:.1f}] m/s²."
        )

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_T = self.r0 + self.v0 * t + 0.5 * self.a_const * (t ** 2)
        v_T = self.v0 + self.a_const * t
        a_T = self.a_const.copy()
        return r_T, v_T, a_T


class CustomFunctionTarget3D(BaseTargetManeuver):
    """
    Arbitrary Mathematical Function Target in 3D Space.
    Accepts explicit time functions:
        x(t) = f(t)
        y(t) = g(t)
        z(t) = h(t)
    Computes velocity and acceleration numerically via O(h^2) central finite differences.
    """

    SAFE_ENV = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
        "np": np,
    }

    def __init__(
        self,
        config: TargetConfig,
        x_expr: str = "6000 - 240*t",
        y_expr: str = "2500 + 350*sin(0.5*t)",
        z_expr: str = "5000 + 200*cos(0.5*t)",
        custom_name: str = "3D Custom Function Target",
    ):
        super().__init__(config)
        self.x_expr = x_expr.strip()
        self.y_expr = y_expr.strip()
        self.z_expr = z_expr.strip()

        # استانداردسازی عملگر توان
        self._clean_x = self.x_expr.replace("^", "**")
        self._clean_y = self.y_expr.replace("^", "**")
        self._clean_z = self.z_expr.replace("^", "**")

        # موقعیت اولیه در t=0
        r0_pos = self._eval_pos(0.0)
        self.r0 = r0_pos
        self.name = f"{custom_name}: x(t)={self.x_expr}, y(t)={self.y_expr}, z(t)={self.z_expr}"
        self.description = (
            f"مسیر پرواز سه‌بعدی با معادلات تحلیلی x(t)={self.x_expr}, y(t)={self.y_expr}, z(t)={self.z_expr}."
        )

    def _eval_pos(self, t_val: float) -> np.ndarray:
        env = dict(self.SAFE_ENV)
        env["t"] = float(t_val)
        try:
            x_val = float(eval(self._clean_x, {"__builtins__": {}}, env))
            y_val = float(eval(self._clean_y, {"__builtins__": {}}, env))
            z_val = float(eval(self._clean_z, {"__builtins__": {}}, env))
        except Exception as e:
            raise ValueError(
                f"خطا در ارزیابی تابع ریاضی هدف سه‌بعدی در t={t_val}:\n"
                f"x={self.x_expr}, y={self.y_expr}, z={self.z_expr}\nپیام خطا: {e}"
            )
        return np.array([x_val, y_val, z_val], dtype=np.float64)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = 1e-5
        r_curr = self._eval_pos(t)
        r_plus = self._eval_pos(t + h)
        r_minus = self._eval_pos(t - h)

        # مشتق‌گیری تفاضل مرکزی
        v_T = (r_plus - r_minus) / (2.0 * h)
        a_T = (r_plus - 2.0 * r_curr + r_minus) / (h ** 2)

        return r_curr, v_T, a_T


TARGET_SCENARIOS: Dict[str, Type[BaseTargetManeuver]] = {
    "A": StraightLineTarget,
    "B": BarrelRollSpiralTarget,
    "C": HighGInclinedTurnTarget,
    "D": HighG3DSTurnTarget,
    "LINE": LinearEquationTarget3D,
    "PARABOLA": ParabolicEquationTarget3D,
    "CUSTOM": CustomFunctionTarget3D,
}


def create_target_maneuver(
    scenario_id: str,
    config: TargetConfig,
    **kwargs,
) -> BaseTargetManeuver:
    """
    Factory helper to instantiate a target maneuver scenario.
    پشتیبانی از سناریوهای A-D و سناریوهای معادلاتی LINE, PARABOLA, CUSTOM.
    """
    scenario_key = scenario_id.upper().strip()
    if scenario_key not in TARGET_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. Available scenarios: {list(TARGET_SCENARIOS.keys())}"
        )
    target_cls = TARGET_SCENARIOS[scenario_key]
    if kwargs:
        return target_cls(config, **kwargs)
    return target_cls(config)
