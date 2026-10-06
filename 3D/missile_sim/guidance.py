"""
3D Proportional Navigation (TPN) and Augmented Proportional Navigation (APN) Guidance Laws.
==========================================================================================

Theoretical Foundations in 3D Euclidean Space:
----------------------------------------------
1. 3D True Proportional Navigation (3D TPN):
   In three dimensions, the relative position vector and relative velocity vector are:
       R_vec = r_T - r_M
       V_rel = v_T - v_M
   The closing velocity (rate of range decrease) is:
       V_c = - d(R)/dt = - (R_vec . V_rel) / |R_vec|
   The line-of-sight (LOS) angular velocity vector in 3D space is defined canonically as:
       Omega_LOS = (R_vec x V_rel) / |R_vec|^2
   The TPN guidance command vector is directed perpendicular to the LOS vector:
       a_cmd_TPN = N * V_c * (Omega_LOS x R_hat)
   where R_hat = R_vec / |R_vec| is the LOS unit vector, and N is the navigation constant (typically 3 to 5, here N = 4.0).

   Physical Dynamics & Phase Lag:
   In 3D non-planar engagements, the LOS rotation vector spans both azimuth and elevation axes.
   When the target executes aggressive maneuvers, TPN relies solely on the kinematic LOS rate
   induced by target displacement. This creates an intrinsic phase lag, requiring the missile
   to pull increasingly violent lateral accelerations in the terminal phase (Endgame), frequently
   leading to G-saturation and larger miss distances.

2. 3D Augmented Proportional Navigation (3D APN):
   To actively counter target maneuvering without endgame phase lag, APN introduces a target
   acceleration feedforward compensation term.
   The target acceleration component normal to the line-of-sight vector is:
       a_T_perp = a_T - (a_T . R_hat) * R_hat
   The resulting APN command acceleration vector is:
       a_cmd_APN = N * V_c * (Omega_LOS x R_hat) + (N / 2) * a_T_perp

   Physical Advantage:
   By directly feeding forward the target's normal maneuver acceleration in the instantaneous 3D maneuver plane,
   APN cancels out the maneuver drift before it accumulates into LOS rate errors.
   This drastically curtails endgame G-demand, eliminates control saturation, and delivers near-zero miss distances.
"""

from enum import Enum
from typing import Tuple, NamedTuple
import numpy as np


class GuidanceLaw(Enum):
    """Available 3D Guidance Law Algorithms."""
    TPN = "3D True Proportional Navigation (TPN)"
    APN = "3D Augmented Proportional Navigation (APN)"
    PP = "3D Pure Pursuit (Nose-to-Target)"


class Kinematics3D(NamedTuple):
    """Container for instantaneous 3D relative kinematics."""
    R_vec: np.ndarray        # Relative position vector [r_T - r_M] (m)
    range_dist: float        # Range |R_vec| (m)
    R_hat: np.ndarray        # LOS unit vector
    V_rel: np.ndarray        # Relative velocity vector [v_T - v_M] (m/s)
    closing_speed: float     # Closing velocity V_c = -dR/dt (m/s)
    omega_los: np.ndarray    # 3D LOS angular rate vector (rad/s)
    omega_los_mag: float     # Magnitude of LOS angular rate |Omega_LOS| (rad/s)


def compute_relative_kinematics(
    r_M: np.ndarray,
    v_M: np.ndarray,
    r_T: np.ndarray,
    v_T: np.ndarray,
) -> Kinematics3D:
    """
    Compute rigorous 3D relative kinematics between missile and target.

    Args:
        r_M: Missile position vector [x, y, z] in meters.
        v_M: Missile velocity vector [vx, vy, vz] in m/s.
        r_T: Target position vector [x, y, z] in meters.
        v_T: Target velocity vector [vx, vy, vz] in m/s.

    Returns:
        Kinematics3D named tuple containing relative vectors and rates.
    """
    R_vec = r_T - r_M
    range_dist = float(np.linalg.norm(R_vec))

    # Guard against singularity at exact zero distance
    if range_dist < 1e-6:
        R_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        closing_speed = 0.0
        omega_los = np.zeros(3, dtype=np.float64)
        omega_los_mag = 0.0
        V_rel = v_T - v_M
        return Kinematics3D(R_vec, range_dist, R_hat, V_rel, closing_speed, omega_los, omega_los_mag)

    R_hat = R_vec / range_dist
    V_rel = v_T - v_M

    # Closing velocity V_c = - d(R)/dt
    closing_speed = float(-np.dot(R_vec, V_rel) / range_dist)

    # 3D LOS angular velocity vector: Omega_LOS = (R x V_rel) / |R|^2
    omega_los = np.cross(R_vec, V_rel) / (range_dist ** 2)
    omega_los_mag = float(np.linalg.norm(omega_los))

    return Kinematics3D(
        R_vec=R_vec,
        range_dist=range_dist,
        R_hat=R_hat,
        V_rel=V_rel,
        closing_speed=closing_speed,
        omega_los=omega_los,
        omega_los_mag=omega_los_mag,
    )


def calculate_3d_tpn(
    kinematics: Kinematics3D,
    N: float = 4.0,
) -> np.ndarray:
    """
    Calculate 3D True Proportional Navigation (TPN) command acceleration vector.

    Formula:
        a_cmd = N * V_c * (Omega_LOS x R_hat)

    Args:
        kinematics: Kinematics3D container.
        N: Navigation constant (default 4.0).

    Returns:
        3D command acceleration vector in m/s^2.
    """
    # Cross product (Omega_LOS x R_hat) lies strictly in the plane of R and V_rel and is normal to R
    los_turn_vector = np.cross(kinematics.omega_los, kinematics.R_hat)
    a_cmd = N * kinematics.closing_speed * los_turn_vector
    return a_cmd


def calculate_3d_apn(
    kinematics: Kinematics3D,
    a_T: np.ndarray,
    N: float = 4.0,
) -> np.ndarray:
    """
    Calculate 3D Augmented Proportional Navigation (APN) command acceleration vector.

    Formula:
        a_cmd = N * V_c * (Omega_LOS x R_hat) + (N / 2) * a_T_perp
    where:
        a_T_perp = a_T - (a_T . R_hat) * R_hat

    Args:
        kinematics: Kinematics3D container.
        a_T: Instantaneous target acceleration vector [ax, ay, az] in m/s^2.
        N: Navigation constant (default 4.0).

    Returns:
        3D command acceleration vector in m/s^2.
    """
    # Base TPN component
    a_cmd_tpn = calculate_3d_tpn(kinematics, N=N)

    # Target acceleration component normal to the line-of-sight
    a_T_los_proj = np.dot(a_T, kinematics.R_hat) * kinematics.R_hat
    a_T_perp = a_T - a_T_los_proj

    # APN augmentation term: (N / 2) * a_T_perp
    a_cmd_apn = a_cmd_tpn + (0.5 * N) * a_T_perp
    return a_cmd_apn


def calculate_3d_pure_pursuit(
    kinematics: Kinematics3D,
    v_M: np.ndarray,
    K_p: float = 4.0,
) -> np.ndarray:
    """
    Calculate 3D Pure Pursuit (PP) command acceleration vector.
    الگوریتم تعقیب محض در فضای سه‌بعدی (3D Pure Pursuit):
    در این روش بردار جهت‌گیری دماغه و سرعت موشک در تمام لحظات مستقیماً به سمت موقعیت هدف هدایت می‌شود.

    فرمولاسیون سه‌بعدی اقلیدسی:
        بردار سرعت موشک: v_M با بردار یکه v_hat = v_M / |v_M|
        بردار یکه خط دید: R_hat = R / |R|
        زاویه انحراف بین دماغه و خط دید: eta = arccos(clip(v_hat . R_hat, -1, 1))
        بردار چرخش انحراف: e_rot = v_hat x R_hat
        بردار سرعت زاویه‌ای چرخش فرمان:
            omega_cmd = Omega_LOS + K_p * eta_vec
        بردار شتاب فرمان جانبی عمود بر سرعت:
            a_cmd = omega_cmd x v_M

    Args:
        kinematics: مقادیر کینماتیک نسبی لحظه‌ای سه‌بعدی Kinematics3D
        v_M: بردار سرعت کنونی موشک [vx, vy, vz]
        K_p: ضریب هدایت و همگرایی دماغه موشک به سمت هدف (پیش‌فرض 4.0)

    Returns:
        بردار شتاب فرمان جانبی سه‌بعدی در راستای چرخش دماغه موشک (m/s^2)
    """
    v_mag = float(np.linalg.norm(v_M))
    if v_mag < 1e-4:
        return np.zeros(3, dtype=np.float64)

    v_hat = v_M / v_mag
    r_hat = kinematics.R_hat

    # زاویه انحراف خط دید و دماغه موشک
    dot_prod = float(np.clip(np.dot(v_hat, r_hat), -1.0, 1.0))
    eta = float(np.arccos(dot_prod))

    # بردار جهت دوران برای منطبق کردن v_hat بر r_hat
    rot_cross = np.cross(v_hat, r_hat)
    sin_eta = float(np.linalg.norm(rot_cross))

    if sin_eta > 1e-6:
        rot_axis = rot_cross / sin_eta
        eta_vec = eta * rot_axis
    else:
        eta_vec = np.zeros(3, dtype=np.float64)

    # سرعت زاویه‌ای چرخش موشک: دنبال کردن چرخش خط دید + جبران خطای انحراف
    omega_cmd = kinematics.omega_los + K_p * eta_vec

    # شتاب جانبی سه‌بعدی کاملاً عمود بر بردار سرعت
    a_cmd = np.cross(omega_cmd, v_M)
    return a_cmd
