"""
2D Proportional Navigation (TPN) and Augmented Proportional Navigation (APN) Guidance Laws.
==========================================================================================

Theoretical Foundations in 2D Planar Space:
-------------------------------------------
1. 2D True Proportional Navigation (2D TPN):
   In the 2D engagement plane (x, y):
       R_vec = r_T - r_M = [Rx, Ry]
       V_rel = v_T - v_M = [V_rel_x, V_rel_y]
       Range R = |R_vec| = sqrt(Rx^2 + Ry^2)
       LOS Unit Vector: R_hat = [Rx / R, Ry / R]
       LOS Normal Unit Vector: n_hat = [-Ry / R, Rx / R]
   Closing velocity:
       V_c = - d(R)/dt = - (R_vec . V_rel) / R
   The line-of-sight (LOS) angular rate scalar is:
       lambda_dot = (Rx * V_rel_y - Ry * V_rel_x) / R^2
   The 2D TPN commanded acceleration vector is normal to the line-of-sight:
       a_cmd_TPN = N * V_c * lambda_dot * n_hat

   Kinematic Phase Lag:
   In planar high-G engagements, when the fighter executes evasive break-turns,
   TPN only reacts to the accumulated line-of-sight rotation produced by target displacement.
   This introduces a fundamental phase lag, causing an acceleration demand spike at endgame
   which leads to G-saturation and miss distance growth.

2. 2D Augmented Proportional Navigation (2D APN):
   APN feeds forward the target's normal acceleration perpendicular to the line-of-sight:
       a_T_perp = a_T - (a_T . R_hat) * R_hat = (a_T . n_hat) * n_hat
   The resulting APN guidance command is:
       a_cmd_APN = N * V_c * lambda_dot * n_hat + (N / 2) * a_T_perp

   Physical Advantage:
   Directly neutralizing the target maneuver acceleration prevents LOS rotation buildup,
   yielding much lower endgame maneuver demands and pinpoint accuracy.
"""

from enum import Enum
from typing import NamedTuple
import numpy as np


class GuidanceLaw2D(Enum):
    """Available 2D Guidance Algorithms."""
    TPN = "2D True Proportional Navigation (TPN)"
    APN = "2D Augmented Proportional Navigation (APN)"
    PP = "2D Pure Pursuit (Nose-to-Target)"


class Kinematics2D(NamedTuple):
    """Container for instantaneous 2D relative kinematics."""
    R_vec: np.ndarray        # [Rx, Ry] (m)
    range_dist: float        # |R_vec| (m)
    R_hat: np.ndarray        # LOS unit vector
    n_hat: np.ndarray        # LOS normal unit vector (rotated +90 deg)
    V_rel: np.ndarray        # [v_T - v_M] (m/s)
    closing_speed: float     # V_c = -dR/dt (m/s)
    lambda_angle: float      # LOS angle (rad)
    lambda_dot: float        # LOS angular rate d(lambda)/dt (rad/s)


def compute_relative_kinematics_2d(
    r_M: np.ndarray,
    v_M: np.ndarray,
    r_T: np.ndarray,
    v_T: np.ndarray,
) -> Kinematics2D:
    """
    Compute 2D relative kinematics between missile and target.
    """
    R_vec = r_T - r_M
    range_dist = float(np.linalg.norm(R_vec))

    if range_dist < 1e-6:
        R_hat = np.array([1.0, 0.0], dtype=np.float64)
        n_hat = np.array([0.0, 1.0], dtype=np.float64)
        V_rel = v_T - v_M
        return Kinematics2D(
            R_vec=R_vec,
            range_dist=range_dist,
            R_hat=R_hat,
            n_hat=n_hat,
            V_rel=V_rel,
            closing_speed=0.0,
            lambda_angle=0.0,
            lambda_dot=0.0,
        )

    R_hat = R_vec / range_dist
    n_hat = np.array([-R_hat[1], R_hat[0]], dtype=np.float64)
    V_rel = v_T - v_M

    closing_speed = float(-np.dot(R_vec, V_rel) / range_dist)
    lambda_angle = float(np.arctan2(R_vec[1], R_vec[0]))
    lambda_dot = float((R_vec[0] * V_rel[1] - R_vec[1] * V_rel[0]) / (range_dist ** 2))

    return Kinematics2D(
        R_vec=R_vec,
        range_dist=range_dist,
        R_hat=R_hat,
        n_hat=n_hat,
        V_rel=V_rel,
        closing_speed=closing_speed,
        lambda_angle=lambda_angle,
        lambda_dot=lambda_dot,
    )


def calculate_2d_tpn(kinematics: Kinematics2D, N: float = 4.0) -> np.ndarray:
    """
    Calculate 2D True Proportional Navigation (TPN) command acceleration vector.
    Formula:
        a_cmd = N * V_c * lambda_dot * n_hat
    """
    return N * kinematics.closing_speed * kinematics.lambda_dot * kinematics.n_hat


def calculate_2d_apn(
    kinematics: Kinematics2D,
    a_T: np.ndarray,
    N: float = 4.0,
) -> np.ndarray:
    """
    Calculate 2D Augmented Proportional Navigation (APN) command acceleration vector.
    Formula:
        a_cmd = N * V_c * lambda_dot * n_hat + (N / 2) * a_T_perp
    """
    a_cmd_tpn = calculate_2d_tpn(kinematics, N=N)
    a_T_perp = np.dot(a_T, kinematics.n_hat) * kinematics.n_hat
    return a_cmd_tpn + (0.5 * N) * a_T_perp


def calculate_2d_pure_pursuit(
    kinematics: Kinematics2D,
    v_M: np.ndarray,
    K_p: float = 4.0,
) -> np.ndarray:
    """
    Calculate 2D Pure Pursuit (PP) command acceleration vector.
    الگوریتم تعقیب محض (Pure Pursuit):
    در این روش دماغه و بردار سرعت موشک در هر لحظه مستقیماً به سمت موقعیت جاری هدف هدایت می‌شود.
    
    فرمولاسیون کینماتیکی:
        بردار سرعت موشک: v_M با زاویه جهت‌گیری gamma_M = atan2(v_M_y, v_M_x)
        زاویه خط دید: lambda = atan2(R_y, R_x)
        خطای زاویه‌ای راستای دماغه با هدف: eta = lambda - gamma_M
        نرخ چرخش زاویه هدایت دماغه: gamma_dot = lambda_dot + K_p * sin(eta)
        بردار شتاب جانبی عمود بر سرعت موشک:
            a_cmd = |v_M| * (lambda_dot + K_p * sin(eta)) * n_hat_vM
        
    Args:
        kinematics: مقادیر کینماتیک نسبی لحظه‌ای Kinematics2D
        v_M: بردار سرعت کنونی موشک [vx, vy]
        K_p: ضریب کنترل چرخش زاویه‌ای دماغه به سمت خط دید (پیش‌فرض 4.0)

    Returns:
        بردار شتاب فرمان جانبی در صفحه دوبعدی (m/s^2) عمود بر بردار سرعت موشک
    """
    v_mag = float(np.linalg.norm(v_M))
    if v_mag < 1e-4:
        return np.zeros(2, dtype=np.float64)

    v_hat = v_M / v_mag
    # بردار یکه عمود بر راستای سرعت موشک (+90 درجه دوران پادساعت‌گرد)
    n_v = np.array([-v_hat[1], v_hat[0]], dtype=np.float64)

    # محاسبه سینوس و کسینوس زاویه انحراف بین دماغه موشک و بردار خط دید
    # sin(eta) = v_hat x R_hat = v_hat_x * R_hat_y - v_hat_y * R_hat_x = dot(R_hat, n_v)
    sin_eta = float(np.dot(kinematics.R_hat, n_v))
    cos_eta = float(np.dot(kinematics.R_hat, v_hat))
    eta = float(np.arctan2(sin_eta, cos_eta))

    # نرخ دوران زاویه‌ای جهت‌گیری موشک برای حفظ دماغه بر روی هدف
    gamma_dot_cmd = kinematics.lambda_dot + K_p * eta

    # شتاب جانبی عمود بر بردار سرعت
    a_cmd = v_mag * gamma_dot_cmd * n_v
    return a_cmd
