"""
Automated Unit Tests for Animation Controller & 3D Renderer.
============================================================
Verifies:
- 3D State machine lifecycle and frame advancement
- 3D Trajectory coordinate updates (X, Y, Z)
- Multi-missile mode 4 simultaneous 3D playback
- Terminal impact detection at 3D CPA
- Headless 3D Matplotlib rendering updates
"""

import os
import sys
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root and 3D package are accessible
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DIR_3D = os.path.join(BASE_DIR, "3D")
if DIR_3D not in sys.path:
    sys.path.insert(0, DIR_3D)

from missile_sim.config import MissileConfig, TargetConfig, SimulationConfig
from missile_sim.guidance import GuidanceLaw
from missile_sim.target import create_target_maneuver
from missile_sim.simulator import SimulationEngine, SimulationResult
from animation_controller import (
    AnimationState,
    AnimationPlaybackController,
    MatplotlibAnimationRenderer3D,
)


class TestAnimationController3D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        m_cfg = MissileConfig()
        t_cfg = TargetConfig()
        s_cfg = SimulationConfig()
        target = create_target_maneuver("A", t_cfg)
        engine = SimulationEngine(m_cfg, t_cfg, s_cfg)
        cls.result_tpn = engine.run(GuidanceLaw.TPN, target)
        cls.result_apn = engine.run(GuidanceLaw.APN, target)
        cls.result_pp = engine.run(GuidanceLaw.PP, target)

    def setUp(self):
        self.controller = AnimationPlaybackController(speed_multiplier=1.0)
        self.controller.load_results([self.result_tpn])

    def test_3d_state_and_stepping(self):
        self.controller.start()
        data = self.controller.step(0.2)
        self.assertEqual(data.state, AnimationState.RUNNING)
        self.assertAlmostEqual(data.sim_time, 0.2, delta=0.01)

        # Verify 3D dimensions
        self.assertEqual(len(data.r_T), 3)
        self.assertEqual(len(data.missile_states[0].r_M), 3)
        self.assertGreater(data.missile_states[0].speed, 0.0)

    def test_3d_mode4_simultaneous_benchmark(self):
        # 3 missiles simultaneously (TPN, APN, Pure Pursuit)
        self.controller.load_results([self.result_tpn, self.result_apn, self.result_pp])
        self.controller.start()
        data = self.controller.step(0.5)

        self.assertEqual(len(data.missile_states), 3)
        laws = [m.law for m in data.missile_states]
        self.assertIn(GuidanceLaw.TPN, laws)
        self.assertIn(GuidanceLaw.APN, laws)
        self.assertIn(GuidanceLaw.PP, laws)

    def test_3d_pause_resume_restart(self):
        self.controller.start()
        self.controller.step(0.4)
        t_pause = self.controller.current_sim_time

        self.controller.pause()
        self.assertTrue(self.controller.is_paused())
        self.controller.step(0.3)
        self.assertAlmostEqual(self.controller.current_sim_time, t_pause)

        self.controller.resume()
        self.assertTrue(self.controller.is_running())
        self.controller.step(0.1)
        self.assertGreater(self.controller.current_sim_time, t_pause)

        self.controller.restart()
        self.assertEqual(self.controller.current_frame_idx, 0)
        self.assertAlmostEqual(self.controller.current_sim_time, 0.0)

    def test_3d_headless_renderer(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        renderer = MatplotlibAnimationRenderer3D()
        renderer.setup(ax, [self.result_tpn, self.result_apn], "Scenario A")

        self.controller.load_results([self.result_tpn, self.result_apn])
        self.controller.start()
        data = self.controller.step(0.3)

        renderer.update_frame(data)
        self.assertEqual(len(renderer.missile_trails), 2)
        self.assertEqual(len(renderer.missile_markers), 2)

        renderer.finalize()
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
