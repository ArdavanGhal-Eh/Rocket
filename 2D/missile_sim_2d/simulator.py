"""
Simulation Engine and Numerical Integrator for 2D Interception.
================================================================
Implements 4th-order Runge-Kutta (RK4) integration with high-precision
Closest Point of Approach (CPA) calculation down to millimeter accuracy in 2D.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from .config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from .guidance import (
    GuidanceLaw2D,
    Kinematics2D,
    compute_relative_kinematics_2d,
    calculate_2d_tpn,
    calculate_2d_apn,
    calculate_2d_pure_pursuit,
)
from .target import BaseTargetManeuver2D
from .missile import Missile2D


@dataclass
class SimulationResult2D:
    """Encapsulates comprehensive 2D trajectory and telemetry results."""

    law: GuidanceLaw2D
    target_name: str
    target_desc: str

    time: np.ndarray
    r_M: np.ndarray           # Shape (N, 2)
    v_M: np.ndarray           # Shape (N, 2)
    speed_M: np.ndarray       # Shape (N,)
    mach_M: np.ndarray        # Shape (N,)
    mass_M: np.ndarray        # Shape (N,)

    r_T: np.ndarray           # Shape (N, 2)
    v_T: np.ndarray           # Shape (N, 2)
    a_T: np.ndarray           # Shape (N, 2)

    range_dist: np.ndarray    # Shape (N,)
    closing_speed: np.ndarray # Shape (N,)
    lambda_dot: np.ndarray    # Shape (N,)

    a_cmd: np.ndarray         # Shape (N, 2)
    a_lateral: np.ndarray     # Shape (N, 2)
    a_lateral_g: np.ndarray   # Shape (N,)
    a_net_g: np.ndarray       # Shape (N,)
    is_saturated: np.ndarray  # Shape (N,) bool
    control_energy: np.ndarray# Cumulative integral of |a_cmd|^2 dt (m^2/s^3)

    miss_distance: float
    intercept_time: float
    final_speed: float
    final_mach: float
    total_control_energy: float
    peak_lateral_g: float
    saturation_percentage: float
    hit_location_target: np.ndarray
    hit_location_missile: np.ndarray


class SimulationEngine2D:
    """Simulates 2D missile engagement using 4th-order Runge-Kutta integration."""

    def __init__(
        self,
        missile_cfg: MissileConfig2D = MissileConfig2D(),
        target_cfg: TargetConfig2D = TargetConfig2D(),
        sim_cfg: SimulationConfig2D = SimulationConfig2D(),
    ):
        self.missile_cfg = missile_cfg
        self.target_cfg = target_cfg
        self.sim_cfg = sim_cfg
        self.missile = Missile2D(missile_cfg)

    def run(
        self,
        guidance_law: GuidanceLaw2D,
        target_maneuver: BaseTargetManeuver2D,
    ) -> SimulationResult2D:
        dt = self.sim_cfg.dt
        t_max = self.sim_cfg.t_max
        speed_of_sound = self.sim_cfg.speed_of_sound

        r_M = self.missile_cfg.r0.copy()
        r_T, v_T, a_T = target_maneuver.get_state(0.0)

        R0_vec = r_T - r_M
        R0_dist = np.linalg.norm(R0_vec)
        R0_hat = R0_vec / R0_dist

        # Launch velocity aligned with initial LOS
        v_M = self.missile_cfg.v0_magnitude * R0_hat

        t_list: List[float] = []
        r_M_list: List[np.ndarray] = []
        v_M_list: List[np.ndarray] = []
        r_T_list: List[np.ndarray] = []
        v_T_list: List[np.ndarray] = []
        a_T_list: List[np.ndarray] = []
        range_list: List[float] = []
        vc_list: List[float] = []
        lambda_dot_list: List[float] = []
        a_cmd_list: List[np.ndarray] = []
        a_lat_list: List[np.ndarray] = []
        a_lat_g_list: List[float] = []
        a_net_g_list: List[float] = []
        sat_list: List[bool] = []
        ctrl_energy_list: List[float] = []
        mass_list: List[float] = []

        cumulative_energy = 0.0
        t = 0.0

        min_range = R0_dist
        cpa_time = 0.0
        cpa_r_M = r_M.copy()
        cpa_r_T = r_T.copy()
        miss_distance_analytic = R0_dist

        prev_range = R0_dist
        max_lat = self.missile_cfg.max_lateral_accel

        def _saturate_cmd(cmd_vec: np.ndarray) -> np.ndarray:
            mag = float(np.linalg.norm(cmd_vec))
            if mag > max_lat:
                return max_lat * (cmd_vec / mag)
            return cmd_vec

        while t <= t_max:
            r_T, v_T, a_T = target_maneuver.get_state(t)
            kinematics = compute_relative_kinematics_2d(r_M, v_M, r_T, v_T)
            rng = kinematics.range_dist

            # Guidance command
            if guidance_law == GuidanceLaw2D.TPN:
                a_cmd_raw = calculate_2d_tpn(kinematics, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.APN:
                a_cmd_raw = calculate_2d_apn(kinematics, a_T, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.PP:
                a_cmd_raw = calculate_2d_pure_pursuit(kinematics, v_M, K_p=self.missile_cfg.nav_ratio)
            else:
                raise ValueError(f"Unknown guidance law: {guidance_law}")

            a_cmd = _saturate_cmd(a_cmd_raw)

            # Dynamics & acceleration
            accel_data = self.missile.compute_accelerations(t, v_M, a_cmd)
            a_net = accel_data["a_net"]
            a_lat = accel_data["a_lateral"]
            a_lat_g = accel_data["a_lateral_mag"] / self.missile_cfg.g_accel
            a_net_g = accel_data["a_net_mag"] / self.missile_cfg.g_accel
            is_sat = (np.linalg.norm(a_cmd_raw) > max_lat) or accel_data["is_saturated"]
            mass_val = accel_data["mass"]

            cmd_sq = float(np.dot(a_cmd, a_cmd))
            cumulative_energy += cmd_sq * dt

            # Logging
            t_list.append(t)
            r_M_list.append(r_M.copy())
            v_M_list.append(v_M.copy())
            r_T_list.append(r_T.copy())
            v_T_list.append(v_T.copy())
            a_T_list.append(a_T.copy())
            range_list.append(rng)
            vc_list.append(kinematics.closing_speed)
            lambda_dot_list.append(kinematics.lambda_dot)
            a_cmd_list.append(a_cmd.copy())
            a_lat_list.append(a_lat.copy())
            a_lat_g_list.append(a_lat_g)
            a_net_g_list.append(a_net_g)
            sat_list.append(is_sat)
            ctrl_energy_list.append(cumulative_energy)
            mass_list.append(mass_val)

            # Continuous inter-step analytical CPA calculation
            v_rel = v_T - v_M
            v_rel_sq = np.dot(v_rel, v_rel)
            if v_rel_sq > 1e-4:
                tau_cpa = -np.dot(kinematics.R_vec, v_rel) / v_rel_sq
                if 0.0 <= tau_cpa <= dt:
                    r_cpa_vec = kinematics.R_vec + v_rel * tau_cpa
                    r_cpa_dist = float(np.linalg.norm(r_cpa_vec))
                    if r_cpa_dist < miss_distance_analytic:
                        miss_distance_analytic = r_cpa_dist
                        cpa_time = t + tau_cpa
                        cpa_r_M = r_M + v_M * tau_cpa
                        cpa_r_T = r_T + v_T * tau_cpa

            if rng < min_range:
                min_range = rng
                cpa_discrete_time = t
                cpa_discrete_r_M = r_M.copy()
                cpa_discrete_r_T = r_T.copy()

            if rng <= 0.5:
                miss_distance_analytic = rng
                cpa_time = t
                cpa_r_M = r_M.copy()
                cpa_r_T = r_T.copy()
                break

            # Geometric flyby: target crossed behind missile velocity vector
            if np.dot(kinematics.R_vec, v_M) <= 0.0 and t > 0.5:
                break

            # Terminal CPA separation: reached close range and separating
            if rng > prev_range and rng < 50.0 and t > 0.5:
                break

            prev_range = rng

            # RK4 Integration
            k1_v = a_net
            k1_r = v_M

            t_half = t + 0.5 * dt
            r_M_half1 = r_M + 0.5 * dt * k1_r
            v_M_half1 = v_M + 0.5 * dt * k1_v
            r_T_half, v_T_half, a_T_half = target_maneuver.get_state(t_half)
            kin_half1 = compute_relative_kinematics_2d(r_M_half1, v_M_half1, r_T_half, v_T_half)
            if guidance_law == GuidanceLaw2D.TPN:
                a_cmd_h1 = calculate_2d_tpn(kin_half1, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.APN:
                a_cmd_h1 = calculate_2d_apn(kin_half1, a_T_half, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.PP:
                a_cmd_h1 = calculate_2d_pure_pursuit(kin_half1, v_M_half1, K_p=self.missile_cfg.nav_ratio)
            a_cmd_h1 = _saturate_cmd(a_cmd_h1)
            k2_v = self.missile.compute_accelerations(t_half, v_M_half1, a_cmd_h1)["a_net"]
            k2_r = v_M_half1

            r_M_half2 = r_M + 0.5 * dt * k2_r
            v_M_half2 = v_M + 0.5 * dt * k2_v
            kin_half2 = compute_relative_kinematics_2d(r_M_half2, v_M_half2, r_T_half, v_T_half)
            if guidance_law == GuidanceLaw2D.TPN:
                a_cmd_h2 = calculate_2d_tpn(kin_half2, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.APN:
                a_cmd_h2 = calculate_2d_apn(kin_half2, a_T_half, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.PP:
                a_cmd_h2 = calculate_2d_pure_pursuit(kin_half2, v_M_half2, K_p=self.missile_cfg.nav_ratio)
            a_cmd_h2 = _saturate_cmd(a_cmd_h2)
            k3_v = self.missile.compute_accelerations(t_half, v_M_half2, a_cmd_h2)["a_net"]
            k3_r = v_M_half2

            t_full = t + dt
            r_M_full = r_M + dt * k3_r
            v_M_full = v_M + dt * k3_v
            r_T_full, v_T_full, a_T_full = target_maneuver.get_state(t_full)
            kin_full = compute_relative_kinematics_2d(r_M_full, v_M_full, r_T_full, v_T_full)
            if guidance_law == GuidanceLaw2D.TPN:
                a_cmd_full = calculate_2d_tpn(kin_full, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.APN:
                a_cmd_full = calculate_2d_apn(kin_full, a_T_full, N=self.missile_cfg.nav_ratio)
            elif guidance_law == GuidanceLaw2D.PP:
                a_cmd_full = calculate_2d_pure_pursuit(kin_full, v_M_full, K_p=self.missile_cfg.nav_ratio)
            a_cmd_full = _saturate_cmd(a_cmd_full)
            k4_v = self.missile.compute_accelerations(t_full, v_M_full, a_cmd_full)["a_net"]
            k4_r = v_M_full

            r_M += (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
            v_M += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

            t += dt

        time_arr = np.array(t_list, dtype=np.float64)
        r_M_arr = np.array(r_M_list, dtype=np.float64)
        v_M_arr = np.array(v_M_list, dtype=np.float64)
        r_T_arr = np.array(r_T_list, dtype=np.float64)
        v_T_arr = np.array(v_T_list, dtype=np.float64)
        a_T_arr = np.array(a_T_list, dtype=np.float64)
        range_arr = np.array(range_list, dtype=np.float64)
        vc_arr = np.array(vc_list, dtype=np.float64)
        ldot_arr = np.array(lambda_dot_list, dtype=np.float64)
        a_cmd_arr = np.array(a_cmd_list, dtype=np.float64)
        a_lat_arr = np.array(a_lat_list, dtype=np.float64)
        a_lat_g_arr = np.array(a_lat_g_list, dtype=np.float64)
        a_net_g_arr = np.array(a_net_g_list, dtype=np.float64)
        sat_arr = np.array(sat_list, dtype=bool)
        ctrl_energy_arr = np.array(ctrl_energy_list, dtype=np.float64)
        mass_arr = np.array(mass_list, dtype=np.float64)

        speed_M_arr = np.linalg.norm(v_M_arr, axis=1)
        mach_M_arr = speed_M_arr / speed_of_sound

        final_speed = float(speed_M_arr[-1])
        final_mach = float(mach_M_arr[-1])
        peak_lat_g = float(np.max(a_lat_g_arr))
        sat_pct = float(np.mean(sat_arr) * 100.0)
        final_miss_dist = min(min_range, miss_distance_analytic)
        if (cpa_time <= 1e-4 or final_miss_dist >= min_range) and len(time_arr) > 0:
            min_idx = int(np.argmin(range_arr))
            cpa_time = float(time_arr[min_idx])
            cpa_r_M = r_M_arr[min_idx].copy()
            cpa_r_T = r_T_arr[min_idx].copy()
            final_miss_dist = float(range_arr[min_idx])

        return SimulationResult2D(
            law=guidance_law,
            target_name=target_maneuver.name,
            target_desc=target_maneuver.description,
            time=time_arr,
            r_M=r_M_arr,
            v_M=v_M_arr,
            speed_M=speed_M_arr,
            mach_M=mach_M_arr,
            mass_M=mass_arr,
            r_T=r_T_arr,
            v_T=v_T_arr,
            a_T=a_T_arr,
            range_dist=range_arr,
            closing_speed=vc_arr,
            lambda_dot=ldot_arr,
            a_cmd=a_cmd_arr,
            a_lateral=a_lat_arr,
            a_lateral_g=a_lat_g_arr,
            a_net_g=a_net_g_arr,
            is_saturated=sat_arr,
            control_energy=ctrl_energy_arr,
            miss_distance=final_miss_dist,
            intercept_time=cpa_time,
            final_speed=final_speed,
            final_mach=final_mach,
            total_control_energy=cumulative_energy,
            peak_lateral_g=peak_lat_g,
            saturation_percentage=sat_pct,
            hit_location_target=cpa_r_T,
            hit_location_missile=cpa_r_M,
        )
