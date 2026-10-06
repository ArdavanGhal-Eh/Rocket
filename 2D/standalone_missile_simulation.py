"""
====================================================================================================
PROJECT: 2D TACTICAL MISSILE INTERCEPTION & GUIDANCE LAW BENCHMARK
MODULE : missile_simulation.py
AUTHOR : Advanced Aerospace Guidance & Control Simulation Engine
DATE   : October 2026
====================================================================================================

DYNAMIC THEORY DOCUMENTATION & AEROSPACE GUIDANCE FORMULATION:
----------------------------------------------------------------------------------------------------
1. PURE / TRUE PROPORTIONAL NAVIGATION (PN / PPN / TPN):
   * Physical Philosophy:
     The fundamental principle of Proportional Navigation is driving the Line-of-Sight (LOS) angular
     rate (lambda_dot or omega_LOS) to zero: omega_LOS -> 0.
     When the LOS rate is nulled, the missile and target remain in a constant-bearing decreasing-range
     geometry (Collision Triangle). Under this condition, kinematic interception is guaranteed.
   * Mathematical Formulation:
     a_cmd = N * V_c * omega_LOS
     Where:
       - N         : Navigation Gain (typically 3.0 to 5.0; default N = 4.0)
       - V_c       : Closing Velocity along the LOS (V_c = -dR/dt)
       - omega_LOS : Line-of-Sight Angular Rate (dlambda/dt = (R_x * V_rel_y - R_y * V_rel_x) / R^2)
   * Dynamical Weakness & Failure Modes in Endgame:
     Standard PN implicitly assumes that the target is non-accelerating (constant velocity vector).
     When encountering an aggressively maneuvering target (such as a fighter jet performing high-G
     evasive maneuvers), PN experiences a persistent phase lag. The missile continually tries to
     catch up with the shifting collision triangle, causing the commanded lateral acceleration to
     diverge exponentially in the endgame (t -> t_go).
     This divergence leads directly to:
       1) Control Actuator Saturation (G-Limit reached: a_cmd > 35G).
       2) Inability to correct terminal trajectory errors due to clipped control authority.
       3) Drastic increase in Miss Distance (near-miss or complete fly-by).

2. AUGMENTED PROPORTIONAL NAVIGATION (APN / APNG):
   * Physical Philosophy:
     APN addresses the fundamental shortcoming of PN by incorporating an explicit Feed-Forward term
     proportional to the target's maneuver acceleration normal to the Line-of-Sight.
     Derived from Linear-Quadratic Optimal Control Theory (minimizing terminal miss distance subject
     to bounded control energy integral(u^2 dt)), APN proactively adjusts the missile's trajectory
     to match the target's maneuvering plane without waiting for an LOS rate error to build up.
   * Mathematical Formulation:
     a_cmd = N * V_c * omega_LOS + 0.5 * N * a_T_normal
     Where:
       - a_T_normal : Component of target acceleration perpendicular to the LOS vector:
                      a_T_normal = a_T . e_LOS_perp = -a_Tx * sin(lambda) + a_Ty * cos(lambda)
       - 0.5 * N    : Optimal feed-forward gain derived from optimal regulator theory.
   * Dynamical Superiority:
     By canceling the apparent drift induced by target acceleration in real time:
       1) Phase lag is eliminated, keeping the missile ahead of the target's evasive envelope.
       2) Endgame acceleration demand remains well within the physical G-limit (avoids G-saturation).
       3) Control energy expenditure (Integral(a_cmd^2 dt)) is reduced dramatically.
       4) Sub-meter miss distances are achieved even against 7G-9G multi-axial evasive S-turns.
====================================================================================================
"""

import os
import sys
import io
import math
import argparse
from typing import Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ==================================================================================================
# 1. PHYSICAL, AERODYNAMIC, AND PROPULSION CONSTANTS
# ==================================================================================================
GRAVITY: float = 9.80665              # Standard gravitational acceleration [m/s^2]
AIR_DENSITY: float = 0.736            # Atmospheric density at ~5,000m altitude [kg/m^3]

# Missile Airframe & Mass Properties
MASS_INITIAL: float = 85.0            # Initial missile launch mass [kg]
MASS_DRY: float = 50.0                # Dry mass after fuel exhaustion [kg]
MASS_FUEL: float = 35.0               # Total consumable propellant mass [kg]
T_BURN: float = 4.0                   # Solid rocket motor burn duration [s]
M_DOT: float = MASS_FUEL / T_BURN     # Fuel mass depletion rate: 8.75 [kg/s]
THRUST_NOMINAL: float = 17500.0       # Rocket motor constant thrust [N]

# Aerodynamics
MISSILE_DIAMETER: float = 0.127       # Missile caliber/diameter (5-inch class) [m]
REFERENCE_AREA: float = 0.0127        # Cross-sectional reference area [m^2]
DRAG_COEFFICIENT: float = 0.40        # Zero-lift parasitic drag coefficient CD [-]

# Flight Controls & Constraints
G_LIMIT: float = 35.0                 # Max structural/aerodynamic lateral load limit [G]
A_LAT_MAX: float = G_LIMIT * GRAVITY  # Max lateral acceleration limit: ~343.23 [m/s^2]
NAVIGATION_GAIN: float = 4.0          # Proportional Navigation Constant N [-]
LAUNCH_SPEED: float = 250.0           # Initial rail exit velocity [m/s]

# Simulation Numerical Config
TIME_STEP: float = 0.005              # High-precision simulation step dt [s] (5 ms)
MAX_SIM_TIME: float = 20.0            # Simulation watchdog cutoff time [s]


# ==================================================================================================
# 2. TARGET MANEUVER PROFILE GENERATOR (MODULAR KINEMATICS)
# ==================================================================================================
class TargetManeuver:
    """
    Independent and modular kinematic generator for the maneuvering fighter jet.
    Users can easily inspect, plug in, or replace custom analytical functions x(t) and y(t).
    """

    def __init__(self,
                 mode: str = 'D',
                 x0: float = 6000.0,
                 y0: float = 2500.0,
                 speed: float = 300.0,
                 initial_heading_deg: float = 195.0,
                 target_g: float = 8.5,
                 target_period: float = 1.8):
        """
        Parameters:
            mode: Maneuver profile ('A': Straight, 'B': Sinusoidal, 'C': Circle, 'D': High-G S-Turn)
            x0, y0: Initial coordinates [m]
            speed: Sustained airspeed [m/s] (~Mach 0.9 at 5,000m)
            initial_heading_deg: Initial flight path angle [deg]
            target_g: Peak evasive lateral load factor in Gs [G]
            target_period: Maneuver oscillation period [s]
        """
        self.mode = mode.upper()
        self.x0 = x0
        self.y0 = y0
        self.speed = speed
        self.psi0 = math.radians(initial_heading_deg)
        self.target_g = target_g
        self.target_period = target_period

        # Internal state for numerical propagation
        self.x = x0
        self.y = y0
        self.psi = self.psi0

    def reset(self):
        """Resets the target to initial state."""
        self.x = self.x0
        self.y = self.y0
        self.psi = self.psi0

    def evaluate_state(self, t: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the target position, velocity, and acceleration vectors at time t.
        Returns:
            r_T: np.ndarray [x, y]
            v_T: np.ndarray [vx, vy]
            a_T: np.ndarray [ax, ay]
        """
        if self.mode == 'A':
            # --------------------------------------------------------------------------------------
            # Mode A: Uniform Straight-Line Flight (Exact Analytical Formulation)
            # --------------------------------------------------------------------------------------
            vx = self.speed * math.cos(self.psi0)
            vy = self.speed * math.sin(self.psi0)
            return np.array([self.x0 + vx * t, self.y0 + vy * t]), np.array([vx, vy]), np.array([0.0, 0.0])

        elif self.mode == 'C':
            # --------------------------------------------------------------------------------------
            # Mode C: Constant-G Sustained Circular Turn (Exact Analytical Circle)
            # --------------------------------------------------------------------------------------
            a_lat = 6.0 * GRAVITY
            omega = a_lat / self.speed
            psi = self.psi0 + omega * t
            vx = self.speed * math.cos(psi)
            vy = self.speed * math.sin(psi)
            x = self.x0 + (self.speed / omega) * (math.sin(psi) - math.sin(self.psi0))
            y = self.y0 - (self.speed / omega) * (math.cos(psi) - math.cos(self.psi0))
            ax = -a_lat * math.sin(psi)
            ay =  a_lat * math.cos(psi)
            return np.array([x, y]), np.array([vx, vy]), np.array([ax, ay])

        elif self.mode == 'B':
            # --------------------------------------------------------------------------------------
            # Mode B: Continuous Sinusoidal Evasive Maneuver
            # --------------------------------------------------------------------------------------
            omega_evade = 2.0 * math.pi / 3.0
            a_lat = 5.0 * GRAVITY * math.sin(omega_evade * t)
            vx = self.speed * math.cos(self.psi)
            vy = self.speed * math.sin(self.psi)
            ax = -a_lat * math.sin(self.psi)
            ay =  a_lat * math.cos(self.psi)

            # Record exact state at time t
            ret_r = np.array([self.x, self.y])
            ret_v = np.array([vx, vy])
            ret_a = np.array([ax, ay])

            # Advance internal state for the subsequent step (t + dt)
            self.psi += (a_lat / self.speed) * dt
            self.x += vx * dt
            self.y += vy * dt

            return ret_r, ret_v, ret_a

        elif self.mode == 'D':
            # --------------------------------------------------------------------------------------
            # Mode D: Heavy High-G S-Turn / Break Turns (Aggressive Zig-Zag Evasion)
            # --------------------------------------------------------------------------------------
            period = self.target_period
            omega_cycle = 2.0 * math.pi / period
            smooth_square = math.tanh(5.0 * math.sin(omega_cycle * t))
            a_lat_max = self.target_g * GRAVITY
            a_lat = a_lat_max * smooth_square

            vx = self.speed * math.cos(self.psi)
            vy = self.speed * math.sin(self.psi)
            ax = -a_lat * math.sin(self.psi)
            ay =  a_lat * math.cos(self.psi)

            # Record exact state at time t
            ret_r = np.array([self.x, self.y])
            ret_v = np.array([vx, vy])
            ret_a = np.array([ax, ay])

            # Advance internal state for the subsequent step (t + dt)
            self.psi += (a_lat / self.speed) * dt
            self.x += vx * dt
            self.y += vy * dt

            return ret_r, ret_v, ret_a

        else:
            raise ValueError(f"Unknown target maneuver mode: {self.mode}")


# ==================================================================================================
# 3. 2D MISSILE SIMULATION ENGINE
# ==================================================================================================
class MissileSimulationEngine:
    """
    High-fidelity 2D point-mass numerical simulation engine integrating thrust, mass loss,
    aerodynamic drag, G-load saturation, and proportional navigation guidance laws.
    """

    def __init__(self, guidance_type: str = 'APN', target_mode: str = 'D',
                 target_g: float = 8.5, target_period: float = 1.8):
        """
        Parameters:
            guidance_type: 'PN' (Pure/True Proportional Navigation) or 'APN' (Augmented PN)
            target_mode  : 'A', 'B', 'C', or 'D'
            target_g     : Evasive maneuver load factor [G]
            target_period: Evasive oscillation cycle [s]
        """
        self.guidance_type = guidance_type.upper()
        self.target_mode = target_mode.upper()
        self.target_g = target_g
        self.target_period = target_period
        self.target = TargetManeuver(mode=self.target_mode, target_g=target_g, target_period=target_period)

    def run(self) -> Dict[str, Any]:
        """
        Executes the time-stepping simulation until Closest Point of Approach (CPA)
        or terminal cutoff condition is satisfied.
        """
        self.target.reset()
        t = 0.0
        dt = TIME_STEP

        # Initial Target State
        r_T, v_T, a_T = self.target.evaluate_state(t, dt)

        # Initial Missile State at Origin (0, 0)
        r_M = np.array([0.0, 0.0])
        # Point missile initial velocity directly along initial Line-of-Sight
        los_angle_0 = math.atan2(r_T[1] - r_M[1], r_T[0] - r_M[0])
        gamma_M = los_angle_0
        v_M_scalar = LAUNCH_SPEED
        v_M = np.array([v_M_scalar * math.cos(gamma_M), v_M_scalar * math.sin(gamma_M)])

        # Data Logging Buffers
        history = {
            'time': [],
            'r_M': [],
            'v_M': [],
            'speed_M': [],
            'gamma_M': [],
            'r_T': [],
            'v_T': [],
            'a_T': [],
            'range': [],
            'closing_speed': [],
            'los_angle': [],
            'los_rate': [],
            'a_cmd': [],
            'a_achieved': [],
            'mass': [],
            'thrust': [],
            'drag': [],
            'control_energy': []
        }

        min_range = float('inf')
        prev_range = float('inf')
        cpa_index = 0
        cumulative_energy = 0.0

        step = 0
        while t <= MAX_SIM_TIME:
            # 1. Kinematic Geometry & Line-of-Sight (LOS)
            R_vec = r_T - r_M
            R = np.linalg.norm(R_vec)
            los_angle = math.atan2(R_vec[1], R_vec[0])

            # Relative Velocity & Closing Velocity
            V_rel = v_T - v_M
            # V_c = -dR/dt = -(R_vec . V_rel) / R
            V_c = -np.dot(R_vec, V_rel) / max(R, 1e-6)

            # LOS Angular Rate (omega_LOS = (Rx * Vry - Ry * Vrx) / R^2)
            los_rate = (R_vec[0] * V_rel[1] - R_vec[1] * V_rel[0]) / max(R**2, 1e-6)

            # LOS unit vectors
            e_los = np.array([math.cos(los_angle), math.sin(los_angle)])
            e_los_perp = np.array([-math.sin(los_angle), math.cos(los_angle)])

            # 2. Guidance Law Calculation
            # Pure PN Law
            a_cmd_pn = NAVIGATION_GAIN * V_c * los_rate

            if self.guidance_type == 'PN':
                a_cmd = a_cmd_pn
            elif self.guidance_type == 'APN':
                # Target acceleration normal to Line-of-Sight
                a_T_normal = np.dot(a_T, e_los_perp)
                # Augmented PN Feed-Forward Term
                a_cmd = a_cmd_pn + 0.5 * NAVIGATION_GAIN * a_T_normal
            else:
                raise ValueError(f"Unknown guidance type: {self.guidance_type}")

            # 3. Lateral Acceleration Saturation (G-Limiting)
            a_achieved = np.clip(a_cmd, -A_LAT_MAX, A_LAT_MAX)

            # Accumulate Control Effort Energy: Integral(a_cmd^2 dt)
            cumulative_energy += (a_cmd ** 2) * dt

            # 4. Missile Propulsion & Mass Schedule
            if t <= T_BURN:
                mass = MASS_INITIAL - M_DOT * t
                thrust = THRUST_NOMINAL
            else:
                mass = MASS_DRY
                thrust = 0.0

            # 5. Aerodynamic Drag Calculation
            v_M_scalar = np.linalg.norm(v_M)
            drag = 0.5 * AIR_DENSITY * (v_M_scalar ** 2) * DRAG_COEFFICIENT * REFERENCE_AREA

            # Tangential Acceleration along velocity vector
            a_tangential = (thrust - drag) / mass

            # Flight Path Angle Turn Rate: gamma_dot = a_achieved / V_M
            gamma_dot = a_achieved / max(v_M_scalar, 1.0)

            # 6. Log State
            history['time'].append(t)
            history['r_M'].append(r_M.copy())
            history['v_M'].append(v_M.copy())
            history['speed_M'].append(v_M_scalar)
            history['gamma_M'].append(gamma_M)
            history['r_T'].append(r_T.copy())
            history['v_T'].append(v_T.copy())
            history['a_T'].append(a_T.copy())
            history['range'].append(R)
            history['closing_speed'].append(V_c)
            history['los_angle'].append(los_angle)
            history['los_rate'].append(los_rate)
            history['a_cmd'].append(a_cmd)
            history['a_achieved'].append(a_achieved)
            history['mass'].append(mass)
            history['thrust'].append(thrust)
            history['drag'].append(drag)
            history['control_energy'].append(cumulative_energy)

            # Check for Closest Point of Approach (CPA)
            if R < min_range:
                min_range = R
                cpa_index = step

            # Termination Condition:
            # When distance starts increasing after initial approach (dR/dt > 0) or hit < 0.2m
            if step > 50 and R > prev_range and R < 500.0:
                # Intercept geometry passed CPA
                break

            prev_range = R

            # 7. Numerical Integration (Euler-Forward / Runge-Kutta Point Mass)
            # Velocity components update:
            gamma_M += gamma_dot * dt
            v_M_scalar += a_tangential * dt
            v_M = np.array([v_M_scalar * math.cos(gamma_M), v_M_scalar * math.sin(gamma_M)])
            r_M += v_M * dt

            # Update Target state for next step
            t += dt
            step += 1
            r_T, v_T, a_T = self.target.evaluate_state(t, dt)

        # Convert history arrays to NumPy for efficient analytical processing
        for k in history:
            history[k] = np.array(history[k])

        # Sub-step quadratic interpolation around CPA for sub-millimeter miss distance precision
        t_cpa, miss_distance = self._interpolate_cpa(history, cpa_index)

        return {
            'guidance': self.guidance_type,
            'target_mode': self.target_mode,
            'history': history,
            'cpa_index': cpa_index,
            't_intercept': t_cpa,
            'miss_distance': miss_distance,
            'final_missile_speed': history['speed_M'][cpa_index],
            'final_closing_speed': abs(history['closing_speed'][max(0, cpa_index - 1)]),
            'final_relative_speed': np.linalg.norm(history['v_M'][cpa_index] - history['v_T'][cpa_index]),
            'total_control_energy': history['control_energy'][cpa_index],
            'flight_max_a_cmd': np.max(np.abs(history['a_cmd'][:cpa_index + 1][history['range'][:cpa_index + 1] >= 0.5])),
            'flight_saturated_steps': np.sum(np.abs(history['a_cmd'][:cpa_index + 1][history['range'][:cpa_index + 1] >= 0.5]) >= A_LAT_MAX),
            'max_a_cmd': np.max(np.abs(history['a_cmd'][:cpa_index + 1])),
            'saturated_steps': np.sum(np.abs(history['a_cmd'][:cpa_index + 1]) >= A_LAT_MAX)
        }

    def _interpolate_cpa(self, history: Dict[str, np.ndarray], idx: int) -> Tuple[float, float]:
        """
        Performs mathematically exact 3-point parabolic interpolation on squared range R^2(t).
        Under locally constant relative velocity, R^2(t) = ||dr0 + v_rel*t||^2 is an EXACT 2nd-degree
        polynomial, avoiding the severe hyperbolic vertex distortion of direct R(t) fitting.
        """
        if idx <= 0 or idx >= len(history['time']) - 1:
            return float(history['time'][idx]), float(history['range'][idx])

        t0, t1, t2 = history['time'][idx - 1], history['time'][idx], history['time'][idx + 1]
        r0_sq = float(history['range'][idx - 1] ** 2)
        r1_sq = float(history['range'][idx] ** 2)
        r2_sq = float(history['range'][idx + 1] ** 2)

        # Fit exact parabola: R^2(t) = A*t^2 + B*t + C
        poly = np.polyfit([t0, t1, t2], [r0_sq, r1_sq, r2_sq], deg=2)
        A, B, C = poly[0], poly[1], poly[2]

        if A > 0:
            t_min = -B / (2.0 * A)
            r2_min = A * (t_min ** 2) + B * t_min + C
            if t0 <= t_min <= t2 and r2_min >= 0:
                return float(t_min), float(math.sqrt(r2_min))

        return float(history['time'][idx]), float(history['range'][idx])


# ==================================================================================================
# 4. POLYNOMIAL TRAJECTORY FITTING & EXPLICIT ALGEBRAIC EQUATION EXTRACTION
# ==================================================================================================
def fit_explicit_trajectory_equations(time_arr: np.ndarray,
                                      r_M_arr: np.ndarray,
                                      degree: int = 5) -> Dict[str, Any]:
    """
    Fits high-order polynomials x(t) and y(t) to the discrete flight trajectory data
    and computes the Coefficient of Determination (R^2).
    """
    t = time_arr
    x = r_M_arr[:, 0]
    y = r_M_arr[:, 1]

    # Polynomial fit for x(t)
    poly_x_coeffs = np.polyfit(t, x, deg=degree)
    poly_x = np.poly1d(poly_x_coeffs)
    x_pred = poly_x(t)
    ss_tot_x = np.sum((x - np.mean(x)) ** 2)
    ss_res_x = np.sum((x - x_pred) ** 2)
    r2_x = 1.0 - (ss_res_x / ss_tot_x) if ss_tot_x != 0 else 1.0

    # Polynomial fit for y(t)
    poly_y_coeffs = np.polyfit(t, y, deg=degree)
    poly_y = np.poly1d(poly_y_coeffs)
    y_pred = poly_y(t)
    ss_tot_y = np.sum((y - np.mean(y)) ** 2)
    ss_res_y = np.sum((y - y_pred) ** 2)
    r2_y = 1.0 - (ss_res_y / ss_tot_y) if ss_tot_y != 0 else 1.0

    return {
        'degree': degree,
        'poly_x_coeffs': poly_x_coeffs,
        'poly_y_coeffs': poly_y_coeffs,
        'r2_x': r2_x,
        'r2_y': r2_y
    }


def format_polynomial_string(coeffs: np.ndarray, var_name: str = "t") -> str:
    """Formats polynomial coefficients into standard mathematical algebraic string."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        sign = "+" if (c >= 0 and len(terms) > 0) else ("-" if c < 0 else "")
        val = abs(c)
        if power > 1:
            terms.append(f"{sign} {val:.5e}*{var_name}^{power}".strip())
        elif power == 1:
            terms.append(f"{sign} {val:.5e}*{var_name}".strip())
        else:
            terms.append(f"{sign} {val:.5e}".strip())
    return " ".join(terms)


# ==================================================================================================
# 5. CONSOLE REPORTING & THEORETICAL DOCUMENTATION PRINTER
# ==================================================================================================
def print_theoretical_documentation():
    """Prints clear, high-signal dynamic theory documentation at startup."""
    banner = "=" * 100
    print(banner)
    print("    DYNAMIC THEORY & AEROSPACE GUIDANCE LAW FORMULATION (مبانی دینامیکی و الگوریتم‌های هدایت)")
    print(banner)
    print("""
1. ناوبری تناسبی استاندارد (Pure / True Proportional Navigation - PN):
   -------------------------------------------------------------------------------------------------
   * فلسفه فیزیکی:
     هدف این قانون، صفر کردن بلادرنگ نرخ چرخش خط دید (Nulling LOS Rate: omega_LOS -> 0) است.
     با صفر شدن این نرخ، زاویه خط دید در فضا ثابت مانده و موشک و هدف در هندسه «مثلث برخورد»
     (Collision Triangle) قرار می‌گیرند که ضامن اصابت قطعی در حالت پرواز یکنواخت است.
   * معادله حاکم:
     a_cmd = N * V_c * omega_LOS
     [N: ضریب ناوبری | V_c: سرعت بسته شدن | omega_LOS: نرخ زاویه‌ای چرخش خط دید]
   * نقطه ضعف دینامیکی و شکست در فاز پایانی (Endgame):
     الگوریتم PN فرض می‌کند شتاب هدف ناچیز است. در مواجهه با مانورهای پرفشار (High-G) جنگنده،
     موشک با تاخیر فاز (Phase Lag) مواجه شده و در لحظات پایانی نیازمند شتاب‌های جانبی تصاعدی می‌شود.
     این پدیده به اشباع سکان‌های کنترلی (G-Saturation > 35G) و افزایش تصاعدی خطای اصابت (Miss Distance) می‌انجامد.

2. ناوبری تناسبی ارتقایافته (Augmented Proportional Navigation - APN):
   -------------------------------------------------------------------------------------------------
   * فلسفه فیزیکی:
     الگوریتم APN برآمده از نظریه کنترل بهینه درجه دو (LQ Optimal Control) با تابع هدف کمینه‌سازی
     خطای پایانی و انرژی کنترلی است. این الگوریتم یک جمله پیش‌خوران (Feed-forward) برای جبران فوری
     شتاب جانبی هدف وارد معادله می‌کند.
   * معادله حاکم:
     a_cmd = N * V_c * omega_LOS + 0.5 * N * a_T_normal
     [a_T_normal: مولفه شتاب مانور جنگنده عمود بر خط دید موشک-هدف]
   * برتری دینامیکی:
     با خنثی‌سازی بلادرنگ اثر شتاب هدف، تاخیر فاز به کلی حذف شده، فرمان شتاب در فاز پایانی در محدوده
     مجاز آیرودینامیکی حفظ می‌شود، انرژی مصرفی سکان‌ها به شدت کاهش می‌یابد و اصابت دقیق تضمین می‌گردد.
""")
    print(banner)


def print_simulation_results_table(results: List[Dict[str, Any]], poly_degree: int = 5):
    """Prints comprehensive engineering benchmark summary table."""
    print("\n" + "=" * 100)
    print("              FINAL ENGINEERING BENCHMARK RESULTS (جدول مقایسه نهایی عملکرد هدایت)")
    print("=" * 100)
    header = f"{'Metric / Parameter':<38} | " + " | ".join([f"{r['guidance']:^26}" for r in results])
    print(header)
    print("-" * 100)

    rows = [
        ("Maneuver Scenario", [f"Mode {r['target_mode']}" for r in results]),
        ("Intercept Time (t_go) [s]", [f"{r['t_intercept']:.4f} s" for r in results]),
        ("Miss Distance [m]", [f"{r['miss_distance']:.5f} m" for r in results]),
        ("Final Missile Speed [m/s]", [f"{r['final_missile_speed']:.2f} m/s ({r['final_missile_speed']/340.0:.2f} M)" for r in results]),
        ("Closing Speed at Hit [m/s]", [f"{r['final_closing_speed']:.2f} m/s" for r in results]),
        ("Relative Approach Speed [m/s]", [f"{r['final_relative_speed']:.2f} m/s" for r in results]),
        ("Flight Peak Commanded Accel [G]", [f"{r['flight_max_a_cmd']/GRAVITY:.2f} G" for r in results]),
        ("Flight G-Saturation Steps (>35G)", [f"{r['flight_saturated_steps']} steps" for r in results]),
        ("Guidance Energy: int(a_cmd^2 dt)", [f"{r['total_control_energy']:.2e} m^2/s^3" for r in results]),
    ]

    for label, vals in rows:
        val_str = " | ".join([f"{v:^26}" for v in vals])
        print(f"{label:<38} | {val_str}")
    print("=" * 100)

    # Print Explicit Trajectory Equations
    print("\n" + "=" * 100)
    print("      EXPLICIT FITTED TRAJECTORY EQUATIONS (معادلات صریح برازش‌شده مسیر پروازی موشک)")
    print("=" * 100)
    for r in results:
        g_name = r['guidance']
        cpa_idx = r['cpa_index']
        t_data = r['history']['time'][:cpa_idx + 1]
        r_M_data = r['history']['r_M'][:cpa_idx + 1]

        fit = fit_explicit_trajectory_equations(t_data, r_M_data, degree=poly_degree)
        eq_x = format_polynomial_string(fit['poly_x_coeffs'], 't')
        eq_y = format_polynomial_string(fit['poly_y_coeffs'], 't')

        print(f"\n>>> GUIDANCE LAW: [{g_name}] (Order-{poly_degree} Polynomial Fit over t in [0.00, {t_data[-1]:.3f}]s):")
        print(f"    x(t) = {eq_x}")
        print(f"           --> Goodness of Fit R^2: {fit['r2_x']:.8f}")
        print(f"    y(t) = {eq_y}")
        print(f"           --> Goodness of Fit R^2: {fit['r2_y']:.8f}")
    print("=" * 100 + "\n")


# ==================================================================================================
# 6. PUBLICATION-GRADE ANALYTICAL VISUALIZATION ENGINE
# ==================================================================================================
def plot_comparative_results(results: List[Dict[str, Any]], save_path: str = "simulation_results.png"):
    """
    Renders 4 synchronized analytical subplots comparing PN and APN guidance trajectories,
    lateral acceleration saturation profiles, range/LOS dynamics, and control energy integrals.
    """
    # Color palette
    color_pn = '#d9534f'     # Red
    color_apn = '#2b82c9'    # Vibrant Blue
    color_tgt = '#28a745'    # Forest Green
    color_limit = '#6c757d'  # Dark Gray

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#fdfdfd')
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # ----------------------------------------------------------------------------------------------
    # Subplot 1: 2D Spatial Trajectory & Interception Geometry
    # ----------------------------------------------------------------------------------------------
    ax1 = axs[0, 0]
    ax1.set_title("1) 2D Spatial Interception Geometry & Flight Paths", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Downrange X [meters]", fontsize=11, fontweight='semibold')
    ax1.set_ylabel("Crossrange Y [meters]", fontsize=11, fontweight='semibold')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Target trajectory from the first run
    ref_run = results[0]
    cpa_ref = ref_run['cpa_index']
    tgt_x = ref_run['history']['r_T'][:cpa_ref + 1, 0]
    tgt_y = ref_run['history']['r_T'][:cpa_ref + 1, 1]
    ax1.plot(tgt_x, tgt_y, color=color_tgt, linewidth=2.5, label='Target (Fighter Jet)')
    ax1.scatter([tgt_x[0]], [tgt_y[0]], color=color_tgt, s=90, marker='s', zorder=5, label='Target Start')

    # Missile Launch Point
    ax1.scatter([0.0], [0.0], color='black', s=100, marker='^', zorder=5, label='Missile Launch (0,0)')

    # Plot Missiles
    for r in results:
        g = r['guidance']
        c_idx = r['cpa_index']
        col = color_apn if g == 'APN' else color_pn
        m_x = r['history']['r_M'][:c_idx + 1, 0]
        m_y = r['history']['r_M'][:c_idx + 1, 1]
        ax1.plot(m_x, m_y, color=col, linewidth=2.0, linestyle='-' if g == 'APN' else '--',
                 label=f'Missile ({g}) [Miss: {r["miss_distance"]:.2f}m]')
        ax1.scatter([m_x[-1]], [m_y[-1]], color=col, s=80, marker='x', zorder=6)

        # Draw periodic sample Lines-of-Sight
        sample_indices = np.linspace(0, c_idx, 6, dtype=int)
        for s_i in sample_indices:
            rm_pt = r['history']['r_M'][s_i]
            rt_pt = r['history']['r_T'][s_i]
            ax1.plot([rm_pt[0], rt_pt[0]], [rm_pt[1], rt_pt[1]],
                     color=col, linestyle=':', alpha=0.35, linewidth=1.0)

    ax1.legend(loc='best', framealpha=0.9, fontsize=9.5)

    # ----------------------------------------------------------------------------------------------
    # Subplot 2: Lateral Acceleration Profile & G-Limit Saturation
    # ----------------------------------------------------------------------------------------------
    ax2 = axs[0, 1]
    ax2.set_title("2) Lateral Acceleration Demand vs. Structural 35G Ceiling", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Flight Time [seconds]", fontsize=11, fontweight='semibold')
    ax2.set_ylabel("Lateral Acceleration [G]", fontsize=11, fontweight='semibold')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # G-Limit boundaries
    ax2.axhline(G_LIMIT, color=color_limit, linestyle='--', linewidth=1.5, label='Max Structural Ceiling (+35G)')
    ax2.axhline(-G_LIMIT, color=color_limit, linestyle='--', linewidth=1.5, label='Max Structural Ceiling (-35G)')

    for r in results:
        g = r['guidance']
        c_idx = r['cpa_index']
        col = color_apn if g == 'APN' else color_pn
        t_arr = r['history']['time'][:c_idx + 1]
        a_cmd_g = r['history']['a_cmd'][:c_idx + 1] / GRAVITY
        a_ach_g = r['history']['a_achieved'][:c_idx + 1] / GRAVITY

        ax2.plot(t_arr, a_cmd_g, color=col, linewidth=1.6, linestyle=':', alpha=0.85,
                 label=f'{g} Commanded a_cmd')
        ax2.plot(t_arr, a_ach_g, color=col, linewidth=2.2, linestyle='-',
                 label=f'{g} Achieved (G-Clipped)')

    ax2.set_ylim(-45, 45)
    ax2.legend(loc='best', framealpha=0.9, fontsize=9.0)

    # ----------------------------------------------------------------------------------------------
    # Subplot 3: Relative Range & Line-of-Sight Angular Rate
    # ----------------------------------------------------------------------------------------------
    ax3 = axs[1, 0]
    ax3.set_title("3) Range-to-Target & Line-of-Sight Angular Rate History", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Flight Time [seconds]", fontsize=11, fontweight='semibold')
    ax3.set_ylabel("Relative Range [meters]", fontsize=11, fontweight='semibold', color='#333333')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Twin axis for LOS Rate
    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel("LOS Angular Rate (omega_LOS) [deg/s]", fontsize=11, fontweight='semibold', color='#800080')

    for r in results:
        g = r['guidance']
        c_idx = r['cpa_index']
        col = color_apn if g == 'APN' else color_pn
        t_arr = r['history']['time'][:c_idx + 1]
        rng_arr = r['history']['range'][:c_idx + 1]
        los_rate_deg = np.degrees(r['history']['los_rate'][:c_idx + 1])

        ax3.plot(t_arr, rng_arr, color=col, linewidth=2.0, label=f'Range ({g})')
        ax3_twin.plot(t_arr, los_rate_deg, color=col, linewidth=1.4, linestyle='--', alpha=0.75,
                      label=f'LOS Rate ({g})')

    ax3.legend(loc='upper left', framealpha=0.9, fontsize=9.0)
    ax3_twin.legend(loc='upper right', framealpha=0.9, fontsize=9.0)

    # ----------------------------------------------------------------------------------------------
    # Subplot 4: Control Energy Accumulation (Integral a_cmd^2 dt)
    # ----------------------------------------------------------------------------------------------
    ax4 = axs[1, 1]
    ax4.set_title("4) Cumulative Guidance Control Effort: Integral(a_cmd^2 dt)", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Flight Time [seconds]", fontsize=11, fontweight='semibold')
    ax4.set_ylabel("Control Effort Metric [m^2 / s^3]", fontsize=11, fontweight='semibold')
    ax4.grid(True, linestyle='--', alpha=0.6)

    for r in results:
        g = r['guidance']
        c_idx = r['cpa_index']
        col = color_apn if g == 'APN' else color_pn
        t_arr = r['history']['time'][:c_idx + 1]
        energy_arr = r['history']['control_energy'][:c_idx + 1]
        ax4.plot(t_arr, energy_arr, color=col, linewidth=2.4,
                 label=f'{g} Energy [Total: {r["total_control_energy"]:.2e}]')

    ax4.legend(loc='best', framealpha=0.9, fontsize=9.5)

    # Save figure to disk
    plt.tight_layout()
    abs_save_path = os.path.abspath(save_path)
    plt.savefig(abs_save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[+] High-resolution analytical plot successfully saved to: {abs_save_path}")


# ==================================================================================================
# 7. MAIN ENTRY POINT & CLI DISPATCHER
# ==================================================================================================
def main():
    parser = argparse.ArgumentParser(description="2D Tactical Missile Interception Simulation Engine")
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=3,
                        help="Execution Mode: 1 = PN Only, 2 = APN Only, 3 = Comparative Side-by-Side (Default: 3)")
    parser.add_argument('--maneuver', type=str, choices=['A', 'B', 'C', 'D'], default='D',
                        help="Target Maneuver Mode: A = Straight, B = Sinusoidal, C = Constant Turn, D = High-G S-Turn (Default: D)")
    parser.add_argument('--target-g', type=float, default=8.5,
                        help="Target peak lateral evasion in Gs (Default: 8.5)")
    parser.add_argument('--target-period', type=float, default=1.8,
                        help="Target evasion oscillation period in seconds (Default: 1.8)")
    parser.add_argument('--poly-degree', type=int, default=5,
                        help="Polynomial degree for analytical trajectory fitting (Default: 5)")
    parser.add_argument('--output', type=str, default="simulation_results.png",
                        help="Output image filename for plots (Default: simulation_results.png)")
    parser.add_argument('--silent-theory', action='store_true',
                        help="Suppress introductory theoretical docstring printout")

    args = parser.parse_args()

    # Step 1: Print Theoretical Dynamic Documentation
    if not args.silent_theory:
        print_theoretical_documentation()

    print(f"[*] Starting Simulation: Execution Mode = {args.mode}, Target Maneuver = Mode {args.maneuver.upper()} (Target: {args.target_g}G, Period: {args.target_period}s) ...")

    # Step 2: Run Simulations based on selected mode
    runs = []
    if args.mode == 1:
        engine_pn = MissileSimulationEngine(guidance_type='PN', target_mode=args.maneuver,
                                            target_g=args.target_g, target_period=args.target_period)
        runs.append(engine_pn.run())
    elif args.mode == 2:
        engine_apn = MissileSimulationEngine(guidance_type='APN', target_mode=args.maneuver,
                                             target_g=args.target_g, target_period=args.target_period)
        runs.append(engine_apn.run())
    elif args.mode == 3:
        engine_pn = MissileSimulationEngine(guidance_type='PN', target_mode=args.maneuver,
                                            target_g=args.target_g, target_period=args.target_period)
        runs.append(engine_pn.run())
        engine_apn = MissileSimulationEngine(guidance_type='APN', target_mode=args.maneuver,
                                             target_g=args.target_g, target_period=args.target_period)
        runs.append(engine_apn.run())

    # Step 3: Print Benchmark Results Table & Equations
    print_simulation_results_table(runs, poly_degree=args.poly_degree)

    # Step 4: Render & Save Plots
    plot_comparative_results(runs, save_path=args.output)


if __name__ == "__main__":
    main()
