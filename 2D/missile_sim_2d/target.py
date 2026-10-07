"""
Modular 2D Target Maneuver Trajectories for Fighter Aircraft.
=============================================================
Provides distinct physical, parametric, and modular 2D planar trajectories:
1. Scenario A: StraightLineTarget2D (Uniform non-maneuvering flight)
2. Scenario B: SinusoidalWeaveTarget2D (Continuous sinusoidal weave oscillation)
3. Scenario C: HighGCircularTurnTarget2D (High-G sustained circular turn)
4. Scenario D: HighG2DSTurnTarget2D (High-G defensive S-turn with smoothed square waves)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type
import numpy as np
from .config import TargetConfig2D
from .safe_math import SafeMathExpression2D, SecurityError


class BaseTargetManeuver2D(ABC):
    """Abstract base class for 2D target trajectories."""

    def __init__(self, config: TargetConfig2D):
        self.config = config
        self.r0 = config.r0.copy()
        self.speed = config.speed
        self.name = "Base Maneuver 2D"
        self.description = "Base parametric trajectory in 2D"

    @abstractmethod
    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute target kinematics at time t.
        Returns (r_T, v_T, a_T) in 2D.
        """
        pass

    def __call__(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_state(t)


class StraightLineTarget2D(BaseTargetManeuver2D):
    """Scenario A: Uniform straight line flight along a constant 2D heading."""

    def __init__(self, config: TargetConfig2D):
        super().__init__(config)
        self.name = "Scenario A: Straight Line Flight"
        self.description = "Uniform, non-accelerating 2D flight along a constant heading."
        self.v_dir = self.config.initial_heading / np.linalg.norm(self.config.initial_heading)
        self.v_const = self.speed * self.v_dir

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_T = self.r0 + self.v_const * t
        v_T = self.v_const.copy()
        a_T = np.zeros(2, dtype=np.float64)
        return r_T, v_T, a_T


class SinusoidalWeaveTarget2D(BaseTargetManeuver2D):
    """
    Scenario B: Continuous sinusoidal evasive weave with ~7.5G normal acceleration.
    """

    def __init__(self, config: TargetConfig2D, period: float = 3.0, g_peak: float = 7.5):
        super().__init__(config)
        self.name = "Scenario B: Sinusoidal Evasive Weave"
        self.description = f"Continuous sinusoidal weave oscillation pulling ~{g_peak:.1f}G normal acceleration."
        self.period = period
        self.omega = 2.0 * np.pi / period
        self.a_peak = g_peak * 9.81

        self.base_dir = self.config.initial_heading / np.linalg.norm(self.config.initial_heading)
        self.norm_dir = np.array([-self.base_dir[1], self.base_dir[0]], dtype=np.float64)

        # Precompute table via RK4 to maintain exact constant speed 300 m/s
        dt_prec = 0.001
        t_max = 30.0
        steps = int(t_max / dt_prec) + 1
        self.dt_prec = dt_prec
        self.times = np.linspace(0, t_max, steps)
        self.r_table = np.zeros((steps, 2), dtype=np.float64)
        self.v_table = np.zeros((steps, 2), dtype=np.float64)
        self.a_table = np.zeros((steps, 2), dtype=np.float64)

        curr_r = self.r0.copy()
        curr_v = self.speed * self.base_dir.copy()

        for i, t_val in enumerate(self.times):
            # Normal commanded acceleration
            a_cmd_mag = self.a_peak * np.sin(self.omega * t_val)
            v_unit = curr_v / np.linalg.norm(curr_v)
            v_perp = np.array([-v_unit[1], v_unit[0]], dtype=np.float64)
            a_normal = a_cmd_mag * v_perp

            self.r_table[i] = curr_r
            self.v_table[i] = curr_v
            self.a_table[i] = a_normal

            curr_r += curr_v * dt_prec
            curr_v += a_normal * dt_prec
            curr_v = self.speed * (curr_v / np.linalg.norm(curr_v))

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class HighGCircularTurnTarget2D(BaseTargetManeuver2D):
    """
    Scenario C: High-G Sustained Circular Turn pulling ~7.5G.
    """

    def __init__(self, config: TargetConfig2D, g_pull: float = 7.5):
        super().__init__(config)
        self.name = "Scenario C: High-G Circular Turn"
        self.description = f"Sustained {g_pull:.1f}G circular turn in the 2D plane."
        self.a_n = g_pull * 9.81
        self.turn_radius = (self.speed ** 2) / self.a_n
        self.omega = self.speed / self.turn_radius

        v0_dir = self.config.initial_heading / np.linalg.norm(self.config.initial_heading)
        u_rad0 = np.array([-v0_dir[1], v0_dir[0]], dtype=np.float64)

        self.u_tan0 = v0_dir
        self.u_rad0 = u_rad0
        self.center = self.r0 + self.turn_radius * self.u_rad0

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.omega * t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        r_T = self.center - self.turn_radius * (cos_t * self.u_rad0 - sin_t * self.u_tan0)
        v_T = self.speed * (sin_t * self.u_rad0 + cos_t * self.u_tan0)
        a_T = self.a_n * (cos_t * self.u_rad0 - sin_t * self.u_tan0)

        return r_T, v_T, a_T


class HighG2DSTurnTarget2D(BaseTargetManeuver2D):
    """
    Scenario D: High-G 2D S-Turn / Break Turns with smoothed square wave profiles (7G-9G).
    """

    def __init__(
        self,
        config: TargetConfig2D,
        period: float = 1.8,
        g_peak: float = 8.5,
        t_precompute: float = 30.0,
        dt_precompute: float = 0.001,
    ):
        super().__init__(config)
        self.name = "Scenario D: High-G 2D S-Turn / Break Turns"
        self.description = (
            f"Violent 2D defensive weave ({g_peak:.1f}G normal) with period {period:.1f}s smoothed square waves."
        )
        self.period = period
        self.omega_s = 2.0 * np.pi / period
        self.a_max = g_peak * 9.81

        base_dir = self.config.initial_heading / np.linalg.norm(self.config.initial_heading)

        self.dt_prec = dt_precompute
        steps = int(t_precompute / dt_precompute) + 1
        self.times = np.linspace(0, t_precompute, steps)
        self.r_table = np.zeros((steps, 2), dtype=np.float64)
        self.v_table = np.zeros((steps, 2), dtype=np.float64)
        self.a_table = np.zeros((steps, 2), dtype=np.float64)

        curr_r = self.r0.copy()
        curr_v = self.speed * base_dir.copy()
        tanh_scale = np.tanh(3.0)

        for i, t_val in enumerate(self.times):
            sq_cmd = np.tanh(3.0 * np.sin(self.omega_s * t_val)) / tanh_scale
            v_norm = np.linalg.norm(curr_v)
            v_unit = curr_v / v_norm
            v_perp = np.array([-v_unit[1], v_unit[0]], dtype=np.float64)
            a_normal = (self.a_max * sq_cmd) * v_perp

            self.r_table[i] = curr_r
            self.v_table[i] = curr_v
            self.a_table[i] = a_normal

            curr_r += curr_v * dt_precompute
            curr_v += a_normal * dt_precompute
            curr_v = self.speed * (curr_v / np.linalg.norm(curr_v))

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class LinearEquationTarget2D(BaseTargetManeuver2D):
    """
    معادله خط هدف در صفحه دوبعدی (Linear Equation Target Trajectory):
    پشتیبانی از هر دو فرم:
    ۱. معادله خط دکارتی: y = m*x + c با سرعت ثابت v_speed و جهت حرکت (direction = -1 به سمت مبدأ)
    ۲. فرم برداری پارامتریک: r_T(t) = r0 + v_const * t
    """

    def __init__(
        self,
        config: TargetConfig2D,
        slope: float = 0.416667,       # شیب خط m (پیش‌فرض y = 2500 در x=6000 به سمت مبدأ)
        intercept: float = 0.0,        # عرض از مبدأ c
        speed: float = 300.0,          # سرعت حرکت در امتداد خط (m/s)
        direction: int = -1,           # جهت بردار سرعت (-1: به سوی مبدأ، +1: موافق محور x)
        x0: float = 6000.0,            # Initial horizontal position
        custom_name: str = "2D Linear Path",
    ):
        super().__init__(config)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.speed = float(speed)
        self.direction = 1 if direction >= 0 else -1
        self.x0 = float(x0)
        self.y0 = self.slope * self.x0 + self.intercept
        self.r0 = np.array([self.x0, self.y0], dtype=np.float64)

        # مولفه‌های سرعت در امتداد خط
        denom = np.sqrt(1.0 + self.slope ** 2)
        vx = self.direction * (self.speed / denom)
        vy = self.slope * vx
        self.v_const = np.array([vx, vy], dtype=np.float64)
        self.a_const = np.zeros(2, dtype=np.float64)

        sign_str = "+" if self.intercept >= 0 else "-"
        self.equation_str = f"y = {self.slope:.4f}x {sign_str} {abs(self.intercept):.1f}"
        self.name = f"{custom_name}: {self.equation_str}"
        self.description = (
            f"پرواز هدف بر روی خط مستقیم {self.equation_str} با سرعت {self.speed:.1f} m/s "
            f"و بردار سرعت [{vx:.1f}, {vy:.1f}] m/s."
        )

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_T = self.r0 + self.v_const * t
        v_T = self.v_const.copy()
        a_T = self.a_const.copy()
        return r_T, v_T, a_T


class ParabolicEquationTarget2D(BaseTargetManeuver2D):
    """
    معادله سهمی هدف در صفحه دوبعدی (Parabolic Equation Target Trajectory):
    پشتیبانی از:
    ۱. سهمی هندسی دکارتی: y(x) = a*x^2 + b*x + c با پیشروی x(t) = x0 + vx*t
    ۲. سهمی پرتابی فیزیکی تحت شتاب ثابت: r_T(t) = r0 + v0*t + 0.5*a_const*t^2
    """

    def __init__(
        self,
        config: TargetConfig2D,
        a: float = 0.00005,            # ضریب درجه دو سهمی y = a*x^2 + b*x + c
        b: float = -0.2,               # ضریب درجه یک
        c: float = 1900.0,             # عدد ثابت
        x0: float = 6000.0,            # موقعیت اولیه x
        vx: float = -260.0,            # Horizontal speed (m/s)
        custom_name: str = "2D Parabolic Path",
    ):
        super().__init__(config)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.x0 = float(x0)
        self.vx = float(vx)
        self.y0 = self.a * (self.x0 ** 2) + self.b * self.x0 + self.c
        self.r0 = np.array([self.x0, self.y0], dtype=np.float64)

        b_sign = "+" if self.b >= 0 else "-"
        c_sign = "+" if self.c >= 0 else "-"
        self.equation_str = f"y = {self.a:.2e}x² {b_sign} {abs(self.b):.3f}x {c_sign} {abs(self.c):.1f}"
        self.name = f"{custom_name}: {self.equation_str}"
        self.description = (
            f"Parabolic target flight based on {self.equation_str} with horizontal speed {self.vx:.1f} m/s."
        )

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Position and time derivatives of parabola
        xt = self.x0 + self.vx * t
        yt = self.a * (xt ** 2) + self.b * xt + self.c

        v_xt = self.vx
        v_yt = (2.0 * self.a * xt + self.b) * self.vx

        a_xt = 0.0
        a_yt = 2.0 * self.a * (self.vx ** 2)

        r_T = np.array([xt, yt], dtype=np.float64)
        v_T = np.array([v_xt, v_yt], dtype=np.float64)
        a_T = np.array([a_xt, a_yt], dtype=np.float64)
        return r_T, v_T, a_T


class CustomFunctionTarget2D(BaseTargetManeuver2D):
    """
    Arbitrary Mathematical Function Target in 2D.
    Accepts explicit time functions:
        x(t) = f(t)
        y(t) = g(t)
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
        config: TargetConfig2D,
        x_expr: str = "6000 - 250*t",
        y_expr: str = "2500 + 400*sin(0.6*t)",
        custom_name: str = "2D Custom Function Target",
    ):
        super().__init__(config)
        self.x_expr = x_expr.strip()
        self.y_expr = y_expr.strip()

        # کامپایل ایمن و بهینه معادلات ریاضی با SafeMathExpression2D
        self.x_parser = SafeMathExpression2D(self.x_expr, variable_name="t")
        self.y_parser = SafeMathExpression2D(self.y_expr, variable_name="t")

        # محاسبه موقعیت اولیه در t=0
        r0_pos = self._eval_pos(0.0)
        self.r0 = r0_pos
        self.name = f"{custom_name}: x(t)={self.x_expr}, y(t)={self.y_expr}"
        self.description = f"مسیر حرکتی هدف با معادلات تحلیلی x(t) = {self.x_expr} و y(t) = {self.y_expr}."

    def _eval_pos(self, t_val: float) -> np.ndarray:
        try:
            x_val = float(self.x_parser.evaluate(t_val))
            y_val = float(self.y_parser.evaluate(t_val))
        except Exception as e:
            raise ValueError(
                f"خطا در ارزیابی تابع ریاضی هدف در t={t_val}:\nx={self.x_expr}, y={self.y_expr}\nپیام خطا: {e}"
            ) from e
        return np.array([x_val, y_val], dtype=np.float64)

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = 1e-5
        r_curr = self._eval_pos(t)
        r_plus = self._eval_pos(t + h)
        r_minus = self._eval_pos(t - h)

        # مشتق‌گیری تفاضل مرکزی مرتبه ۲
        v_T = (r_plus - r_minus) / (2.0 * h)
        a_T = (r_plus - 2.0 * r_curr + r_minus) / (h ** 2)

        return r_curr, v_T, a_T


TARGET_SCENARIOS_2D: Dict[str, Type[BaseTargetManeuver2D]] = {
    "A": StraightLineTarget2D,
    "B": SinusoidalWeaveTarget2D,
    "C": HighGCircularTurnTarget2D,
    "D": HighG2DSTurnTarget2D,
    "LINE": LinearEquationTarget2D,
    "PARABOLA": ParabolicEquationTarget2D,
    "CUSTOM": CustomFunctionTarget2D,
}


def create_target_maneuver_2d(
    scenario_id: str,
    config: TargetConfig2D,
    **kwargs,
) -> BaseTargetManeuver2D:
    """
    Factory helper to instantiate a 2D target maneuver scenario.
    پشتیبانی از سناریوهای استاندارد A-D و سناریوهای معادلاتی LINE, PARABOLA, CUSTOM.
    """
    scenario_key = scenario_id.upper().strip()
    if scenario_key not in TARGET_SCENARIOS_2D:
        raise ValueError(
            f"Unknown scenario '{scenario_id}'. Available scenarios: {list(TARGET_SCENARIOS_2D.keys())}"
        )
    target_cls = TARGET_SCENARIOS_2D[scenario_key]
    if kwargs:
        return target_cls(config, **kwargs)
    return target_cls(config)
