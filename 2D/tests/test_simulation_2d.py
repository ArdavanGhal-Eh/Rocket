import os
import sys
import unittest
import numpy as np

# Ensure 2D directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from missile_sim_2d.config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from missile_sim_2d.guidance import (
    GuidanceLaw2D,
    compute_relative_kinematics_2d,
    calculate_2d_tpn,
    calculate_2d_apn,
    calculate_2d_pure_pursuit,
)
from missile_sim_2d.missile import Missile2D
from missile_sim_2d.target import (
    StraightLineTarget2D,
    LinearEquationTarget2D,
    ParabolicEquationTarget2D,
    CustomFunctionTarget2D,
    create_target_maneuver_2d,
)
from missile_sim_2d.simulator import SimulationEngine2D
from missile_sim_2d.fitting import TrajectoryFitter2D


class TestMissileSimulation2D(unittest.TestCase):

    def setUp(self):
        self.m_cfg = MissileConfig2D()
        self.t_cfg = TargetConfig2D()
        self.s_cfg = SimulationConfig2D()
        self.missile = Missile2D(self.m_cfg)

    def test_physical_parameters(self):
        self.assertEqual(self.m_cfg.m0, 85.0)
        self.assertEqual(self.m_cfg.m_dry, 50.0)
        self.assertEqual(self.m_cfg.t_burn, 4.0)
        self.assertEqual(self.m_cfg.thrust_nominal, 17500.0)
        self.assertEqual(self.m_cfg.mdot, 8.75)
        self.assertEqual(self.m_cfg.rho, 0.736)
        self.assertEqual(self.m_cfg.g_limit_lateral, 35.0)
        self.assertEqual(self.t_cfg.speed, 300.0)
        np.testing.assert_allclose(self.m_cfg.r0, np.array([0.0, 0.0]))
        np.testing.assert_allclose(self.t_cfg.r0, np.array([6000.0, 2500.0]))

    def test_mass_and_thrust(self):
        self.assertAlmostEqual(self.missile.get_mass(0.0), 85.0)
        self.assertAlmostEqual(self.missile.get_mass(4.0), 50.0)
        self.assertAlmostEqual(self.missile.get_mass(6.0), 50.0)

        v_unit = np.array([0.8, 0.6])
        np.testing.assert_allclose(self.missile.get_thrust(2.0, v_unit), 17500.0 * v_unit)
        np.testing.assert_allclose(self.missile.get_thrust(4.5, v_unit), np.zeros(2))

    def test_lateral_g_saturation(self):
        v_unit = np.array([1.0, 0.0])
        a_cmd = np.array([0.0, 80.0 * 9.81])
        sat_a, is_sat = self.missile.enforce_g_limit(a_cmd, v_unit)
        self.assertTrue(is_sat)
        self.assertAlmostEqual(np.linalg.norm(sat_a), 35.0 * 9.81, places=4)

    def test_kinematics_and_guidance(self):
        r_M = np.array([0.0, 0.0])
        v_M = np.array([500.0, 0.0])
        r_T = np.array([2000.0, 500.0])
        v_T = np.array([0.0, 300.0])

        kin = compute_relative_kinematics_2d(r_M, v_M, r_T, v_T)
        self.assertTrue(kin.closing_speed > 0)

        a_cmd_tpn = calculate_2d_tpn(kin, N=4.0)
        dot_los = np.dot(a_cmd_tpn, kin.R_hat)
        self.assertAlmostEqual(dot_los, 0.0, places=5)

        a_T = np.array([0.0, 50.0])
        a_cmd_apn = calculate_2d_apn(kin, a_T, N=4.0)
        self.assertFalse(np.allclose(a_cmd_apn, a_cmd_tpn))

    def test_target_speeds(self):
        for scen_id in ["A", "B", "C", "D"]:
            tgt = create_target_maneuver_2d(scen_id, self.t_cfg)
            for t in [0.0, 1.5, 4.0]:
                r, v, a = tgt.get_state(t)
                speed = np.linalg.norm(v)
                self.assertAlmostEqual(speed, 300.0, delta=1.5)

    def test_simulation_intercept(self):
        engine = SimulationEngine2D(self.m_cfg, self.t_cfg, self.s_cfg)
        tgt = StraightLineTarget2D(self.t_cfg)
        res_tpn = engine.run(GuidanceLaw2D.TPN, tgt)
        res_apn = engine.run(GuidanceLaw2D.APN, tgt)

        self.assertLess(res_tpn.miss_distance, 0.05)
        self.assertLess(res_apn.miss_distance, 0.05)

    def test_pure_pursuit_guidance_2d(self):
        r_M = np.array([0.0, 0.0])
        v_M = np.array([250.0, 0.0])
        r_T = np.array([3000.0, 1000.0])
        v_T = np.array([-250.0, 0.0])

        kin = compute_relative_kinematics_2d(r_M, v_M, r_T, v_T)
        a_cmd = calculate_2d_pure_pursuit(kin, v_M, K_p=4.0)

        # Pure Pursuit commands lateral acceleration normal to velocity
        dot_v = np.dot(a_cmd, v_M)
        self.assertAlmostEqual(dot_v, 0.0, places=5)
        # Heading error is positive, so normal acceleration should turn missile towards positive y
        self.assertGreater(a_cmd[1], 0.0)

    def test_linear_equation_target_2d(self):
        tgt = LinearEquationTarget2D(self.t_cfg, slope=0.5, intercept=500.0, speed=300.0)
        r0, v0, a0 = tgt.get_state(0.0)
        self.assertAlmostEqual(r0[1], 0.5 * r0[0] + 500.0, places=4)
        self.assertAlmostEqual(np.linalg.norm(v0), 300.0, places=4)
        self.assertAlmostEqual(np.linalg.norm(a0), 0.0, places=4)

    def test_parabolic_equation_target_2d(self):
        tgt = ParabolicEquationTarget2D(self.t_cfg, a=0.0001, b=-0.1, c=1500.0, vx=-250.0)
        r0, v0, a0 = tgt.get_state(0.0)
        r2, v2, a2 = tgt.get_state(2.0)
        self.assertAlmostEqual(r0[1], 0.0001 * (r0[0]**2) - 0.1 * r0[0] + 1500.0, places=4)
        self.assertNotEqual(r2[1], r0[1])
        self.assertGreater(abs(a0[1]), 0.0)

    def test_custom_equation_target_2d(self):
        tgt = CustomFunctionTarget2D(self.t_cfg, x_expr="6000 - 250*t", y_expr="2500 + 400*sin(0.5*t)")
        r0, v0, a0 = tgt.get_state(0.0)
        self.assertAlmostEqual(r0[0], 6000.0, places=4)
        self.assertAlmostEqual(r0[1], 2500.0, places=4)
        self.assertAlmostEqual(v0[0], -250.0, places=2)
        self.assertAlmostEqual(v0[1], 200.0, places=2) # 400 * 0.5 * cos(0) = 200

    def test_pure_pursuit_simulation_intercept(self):
        engine = SimulationEngine2D(self.m_cfg, self.t_cfg, self.s_cfg)
        tgt_line = LinearEquationTarget2D(self.t_cfg)
        res_pp = engine.run(GuidanceLaw2D.PP, tgt_line)
        self.assertLess(res_pp.miss_distance, 0.05)
        self.assertGreater(res_pp.final_speed, 900.0)


if __name__ == "__main__":
    unittest.main()
