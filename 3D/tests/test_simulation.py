import os
import sys
import unittest
import numpy as np

# Ensure 3D directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from missile_sim.config import MissileConfig, TargetConfig, SimulationConfig
from missile_sim.guidance import (
    GuidanceLaw,
    compute_relative_kinematics,
    calculate_3d_tpn,
    calculate_3d_apn,
    calculate_3d_pure_pursuit,
)
from missile_sim.missile import Missile3D
from missile_sim.target import (
    StraightLineTarget,
    BarrelRollSpiralTarget,
    HighGInclinedTurnTarget,
    HighG3DSTurnTarget,
    LinearEquationTarget3D,
    ParabolicEquationTarget3D,
    CustomFunctionTarget3D,
    create_target_maneuver,
)
from missile_sim.simulator import SimulationEngine
from missile_sim.fitting import TrajectoryFitter


class TestMissileSimulation(unittest.TestCase):
    """Test suite verifying simulation fidelity and requirement compliance."""

    def setUp(self):
        self.m_cfg = MissileConfig()
        self.t_cfg = TargetConfig()
        self.s_cfg = SimulationConfig()
        self.missile = Missile3D(self.m_cfg)

    def test_physical_parameters(self):
        """Verify all exact requested physical and industrial values."""
        self.assertEqual(self.m_cfg.m0, 85.0)
        self.assertEqual(self.m_cfg.m_dry, 50.0)
        self.assertEqual(self.m_cfg.t_burn, 4.0)
        self.assertEqual(self.m_cfg.thrust_nominal, 17500.0)
        self.assertEqual(self.m_cfg.mdot, 8.75)
        self.assertEqual(self.m_cfg.rho, 0.736)
        self.assertEqual(self.m_cfg.diameter, 0.127)
        self.assertEqual(self.m_cfg.ref_area, 0.0127)
        self.assertEqual(self.m_cfg.cd_default, 0.40)
        self.assertEqual(self.m_cfg.g_limit_lateral, 35.0)
        self.assertEqual(self.m_cfg.v0_magnitude, 250.0)
        self.assertEqual(self.m_cfg.nav_ratio, 4.0)
        self.assertEqual(self.s_cfg.dt, 0.005)
        np.testing.assert_array_equal(self.t_cfg.r0, np.array([6000.0, 2500.0, 5000.0]))
        self.assertEqual(self.t_cfg.speed, 300.0)

    def test_mass_depletion(self):
        """Verify mass behavior during and after solid rocket burn."""
        self.assertAlmostEqual(self.missile.get_mass(0.0), 85.0)
        self.assertAlmostEqual(self.missile.get_mass(2.0), 85.0 - 2.0 * 8.75)
        self.assertAlmostEqual(self.missile.get_mass(4.0), 50.0)
        self.assertAlmostEqual(self.missile.get_mass(4.1), 50.0)
        self.assertAlmostEqual(self.missile.get_mass(10.0), 50.0)

    def test_thrust_profile(self):
        """Verify thrust direction and cutoff after burn time."""
        v_unit = np.array([0.6, 0.8, 0.0])
        t_vec_burn = self.missile.get_thrust(2.0, v_unit)
        np.testing.assert_allclose(t_vec_burn, 17500.0 * v_unit)

        t_vec_coast = self.missile.get_thrust(4.05, v_unit)
        np.testing.assert_allclose(t_vec_coast, np.zeros(3))

    def test_spherical_g_limit_saturation(self):
        """Verify that lateral acceleration commands never exceed 35G."""
        v_unit = np.array([1.0, 0.0, 0.0])
        # Commanded 100G perpendicular to velocity
        a_cmd_huge = np.array([0.0, 100.0 * 9.81, 0.0])
        sat_a, is_sat = self.missile.enforce_spherical_g_limit(a_cmd_huge, v_unit)

        self.assertTrue(is_sat)
        self.assertAlmostEqual(np.linalg.norm(sat_a), 35.0 * 9.81, places=4)
        self.assertAlmostEqual(np.dot(sat_a, v_unit), 0.0, places=6)

    def test_kinematics_and_guidance(self):
        """Verify 3D LOS rate and guidance command formulations."""
        r_M = np.array([0.0, 0.0, 0.0])
        v_M = np.array([500.0, 0.0, 0.0])
        r_T = np.array([1000.0, 100.0, 0.0])
        v_T = np.array([0.0, 200.0, 0.0])

        kin = compute_relative_kinematics(r_M, v_M, r_T, v_T)
        self.assertTrue(kin.closing_speed > 0)
        self.assertTrue(kin.omega_los_mag > 0)

        # Check TPN command is strictly perpendicular to LOS
        a_cmd_tpn = calculate_3d_tpn(kin, N=4.0)
        dot_los = np.dot(a_cmd_tpn, kin.R_hat)
        self.assertAlmostEqual(dot_los, 0.0, places=5)

        # Check APN command
        a_T = np.array([0.0, 50.0, 30.0])
        a_cmd_apn = calculate_3d_apn(kin, a_T, N=4.0)
        # Difference between APN and TPN should equal (N/2) * a_T_perp
        a_T_perp = a_T - np.dot(a_T, kin.R_hat) * kin.R_hat
        expected_diff = 2.0 * a_T_perp
        np.testing.assert_allclose(a_cmd_apn - a_cmd_tpn, expected_diff)

    def test_all_target_maneuvers_speeds(self):
        """Verify that all target scenarios maintain aircraft speed of 300 m/s."""
        times = [0.0, 1.0, 3.5, 7.2]
        for scen_id in ["A", "B", "C", "D"]:
            tgt = create_target_maneuver(scen_id, self.t_cfg)
            for t in times:
                r_T, v_T, a_T = tgt.get_state(t)
                speed = np.linalg.norm(v_T)
                self.assertAlmostEqual(speed, 300.0, delta=1.5, msg=f"Scenario {scen_id} failed speed at t={t}")

    def test_intercept_convergence(self):
        """Verify that simulation successfully intercepts target with miss distance < 0.1 m."""
        engine = SimulationEngine(self.m_cfg, self.t_cfg, self.s_cfg)
        tgt = StraightLineTarget(self.t_cfg)
        res_tpn = engine.run(GuidanceLaw.TPN, tgt)
        res_apn = engine.run(GuidanceLaw.APN, tgt)

        self.assertLess(res_tpn.miss_distance, 0.05)
        self.assertLess(res_apn.miss_distance, 0.05)
        self.assertTrue(res_tpn.intercept_time > 0)
        self.assertTrue(res_apn.intercept_time > 0)

    def test_trajectory_fitter_accuracy(self):
        """Verify polynomial regression satisfies R^2 >= 0.999."""
        t = np.linspace(0, 8.0, 1000)
        x = 200.0 * t + 10.0 * (t ** 2) - 0.5 * (t ** 3)
        y = 50.0 * t + 5.0 * np.sin(0.4 * t)
        z = 300.0 * t - 4.905 * (t ** 2) + 2.0 * (t ** 4) / 100.0
        pos = np.column_stack([x, y, z])

        fitter = TrajectoryFitter(degree=6)
        fit_res = fitter.fit(t, pos)
        self.assertGreater(fit_res.overall_mean_r2, 0.999)
        self.assertGreater(fit_res.x_fit.r2_score, 0.999)
        self.assertGreater(fit_res.y_fit.r2_score, 0.999)
        self.assertGreater(fit_res.z_fit.r2_score, 0.999)

    def test_pure_pursuit_guidance_3d(self):
        r_M = np.array([0.0, 0.0, 0.0])
        v_M = np.array([250.0, 0.0, 0.0])
        r_T = np.array([3000.0, 1000.0, 500.0])
        v_T = np.array([-250.0, 0.0, 0.0])

        kin = compute_relative_kinematics(r_M, v_M, r_T, v_T)
        a_cmd = calculate_3d_pure_pursuit(kin, v_M, K_p=4.0)

        # Commanded acceleration must be strictly normal to velocity
        dot_v = float(np.dot(a_cmd, v_M))
        self.assertAlmostEqual(dot_v, 0.0, places=4)
        self.assertGreater(np.linalg.norm(a_cmd), 0.0)

    def test_linear_equation_target_3d(self):
        tgt = LinearEquationTarget3D(self.t_cfg, speed=300.0)
        r0, v0, a0 = tgt.get_state(0.0)
        self.assertAlmostEqual(np.linalg.norm(v0), 300.0, places=4)
        self.assertAlmostEqual(np.linalg.norm(a0), 0.0, places=4)

    def test_parabolic_equation_target_3d(self):
        tgt = ParabolicEquationTarget3D(self.t_cfg, a_const=np.array([0.0, 10.0, -9.81]))
        r0, v0, a0 = tgt.get_state(0.0)
        r2, v2, a2 = tgt.get_state(2.0)
        self.assertAlmostEqual(a0[2], -9.81, places=4)
        self.assertNotEqual(r2[2], r0[2])

    def test_custom_equation_target_3d(self):
        tgt = CustomFunctionTarget3D(
            self.t_cfg,
            x_expr="6000 - 240*t",
            y_expr="2500 + 350*sin(0.5*t)",
            z_expr="5000 + 200*cos(0.5*t)",
        )
        r0, v0, a0 = tgt.get_state(0.0)
        self.assertAlmostEqual(r0[0], 6000.0, places=4)
        self.assertAlmostEqual(r0[1], 2500.0, places=4)
        self.assertAlmostEqual(r0[2], 5200.0, places=4) # 5000 + 200*cos(0) = 5200
        self.assertAlmostEqual(v0[0], -240.0, places=2)

    def test_pure_pursuit_simulation_intercept_3d(self):
        engine = SimulationEngine(self.m_cfg, self.t_cfg, self.s_cfg)
        head_on = -self.t_cfg.r0 / np.linalg.norm(self.t_cfg.r0)
        tgt = LinearEquationTarget3D(self.t_cfg, heading_dir=head_on)
        res_pp = engine.run(GuidanceLaw.PP, tgt)
        self.assertLess(res_pp.miss_distance, 0.10)
        self.assertGreater(res_pp.final_speed, 900.0)


if __name__ == "__main__":
    unittest.main()
