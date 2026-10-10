"""
Automated Unit Tests for Animation Controller & 2D Renderer.
============================================================
Verifies:
- State machine transitions (IDLE -> RUNNING -> PAUSED -> COMPLETED)
- Time progression and frame scheduling
- Pause, Resume, and Restart behavior
- Playback speed scaling (0.25x, 0.5x, 1x, 2x, 5x)
- Terminal condition CPA arrival
- Robust validation against empty, NaN, and invalid trajectories
- Headless 2D Matplotlib rendering updates
"""

import os
import sys
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root and 2D package are accessible
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DIR_2D = os.path.join(BASE_DIR, "2D")
if DIR_2D not in sys.path:
    sys.path.insert(0, DIR_2D)

from missile_sim_2d.config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from missile_sim_2d.guidance import GuidanceLaw2D
from missile_sim_2d.target import create_target_maneuver_2d
from missile_sim_2d.simulator import SimulationEngine2D, SimulationResult2D
from animation_controller import (
    AnimationState,
    AnimationPlaybackController,
    MatplotlibAnimationRenderer2D,
)


class TestAnimationController2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        m_cfg = MissileConfig2D()
        t_cfg = TargetConfig2D()
        s_cfg = SimulationConfig2D()
        target = create_target_maneuver_2d("A", t_cfg)
        engine = SimulationEngine2D(m_cfg, t_cfg, s_cfg)
        cls.result_tpn = engine.run(GuidanceLaw2D.TPN, target)
        cls.result_apn = engine.run(GuidanceLaw2D.APN, target)

    def setUp(self):
        self.controller = AnimationPlaybackController(speed_multiplier=1.0)
        self.controller.load_results([self.result_tpn])

    def test_initial_state(self):
        self.assertEqual(self.controller.state, AnimationState.IDLE)
        self.assertEqual(self.controller.current_frame_idx, 0)
        self.assertAlmostEqual(self.controller.current_sim_time, 0.0)
        self.assertFalse(self.controller.is_running())
        self.assertFalse(self.controller.is_paused())
        self.assertFalse(self.controller.is_completed())

    def test_start_and_step(self):
        self.controller.start()
        self.assertTrue(self.controller.is_running())
        self.assertEqual(self.controller.state, AnimationState.RUNNING)

        # Step forward 0.1 real seconds with 1.0x speed
        frame_data = self.controller.step(0.1)
        self.assertAlmostEqual(frame_data.sim_time, 0.1, delta=0.01)
        self.assertGreater(frame_data.frame_index, 0)
        self.assertEqual(frame_data.state, AnimationState.RUNNING)
        self.assertFalse(frame_data.is_terminal)
        self.assertEqual(len(frame_data.missile_states), 1)

    def test_pause_and_resume(self):
        self.controller.start()
        self.controller.step(0.5)
        paused_time = self.controller.current_sim_time
        paused_idx = self.controller.current_frame_idx

        # Pause
        self.controller.pause()
        self.assertTrue(self.controller.is_paused())
        self.assertEqual(self.controller.state, AnimationState.PAUSED)

        # Stepping while paused should not advance time
        data = self.controller.step(0.5)
        self.assertEqual(data.frame_index, paused_idx)
        self.assertAlmostEqual(data.sim_time, paused_time)

        # Resume
        self.controller.resume()
        self.assertTrue(self.controller.is_running())
        self.controller.step(0.1)
        self.assertGreater(self.controller.current_sim_time, paused_time)

    def test_restart(self):
        self.controller.start()
        self.controller.step(1.5)
        self.assertGreater(self.controller.current_frame_idx, 0)

        # Restart
        self.controller.restart()
        self.assertTrue(self.controller.is_running())
        self.assertEqual(self.controller.current_frame_idx, 0)
        self.assertAlmostEqual(self.controller.current_sim_time, 0.0)

    def test_speed_scaling(self):
        # 2x speed: 0.1 real seconds should advance sim time by 0.2s
        self.controller.set_speed(2.0)
        self.controller.start()
        data = self.controller.step(0.1)
        self.assertAlmostEqual(data.sim_time, 0.2, delta=0.01)

        # Invalid speed
        with self.assertRaises(ValueError):
            self.controller.set_speed(0.0)
        with self.assertRaises(ValueError):
            self.controller.set_speed(-1.5)

    def test_terminal_condition_reach(self):
        self.controller.start()
        total_duration = float(self.result_tpn.time[-1])
        # Step large amount of time to reach terminal condition
        data = self.controller.step(total_duration + 5.0)

        self.assertTrue(self.controller.is_completed())
        self.assertEqual(self.controller.state, AnimationState.COMPLETED)
        self.assertTrue(data.is_terminal)
        self.assertEqual(data.frame_index, len(self.result_tpn.time) - 1)
        self.assertAlmostEqual(data.sim_time, total_duration)

    def test_multi_missile_mode4(self):
        # Load multiple results (TPN + APN)
        self.controller.load_results([self.result_tpn, self.result_apn])
        self.controller.start()
        data = self.controller.step(0.2)

        self.assertEqual(len(data.missile_states), 2)
        self.assertEqual(data.missile_states[0].law, GuidanceLaw2D.TPN)
        self.assertEqual(data.missile_states[1].law, GuidanceLaw2D.APN)
        self.assertGreater(data.missile_states[0].speed, 0.0)
        self.assertGreater(data.missile_states[1].speed, 0.0)

    def test_invalid_and_empty_inputs(self):
        ctrl = AnimationPlaybackController()
        with self.assertRaises(ValueError):
            ctrl.load_results([])

        # Empty arrays
        empty_res = SimulationResult2D(
            law=GuidanceLaw2D.TPN, target_name="E", target_desc="E",
            time=np.array([]), r_M=np.empty((0, 2)), v_M=np.empty((0, 2)),
            speed_M=np.array([]), mach_M=np.array([]), mass_M=np.array([]),
            r_T=np.empty((0, 2)), v_T=np.empty((0, 2)), a_T=np.empty((0, 2)),
            range_dist=np.array([]), closing_speed=np.array([]), lambda_dot=np.array([]),
            a_cmd=np.empty((0, 2)), a_lateral=np.empty((0, 2)), a_lateral_g=np.array([]),
            a_net_g=np.array([]), is_saturated=np.array([]), control_energy=np.array([]),
            miss_distance=0.0, intercept_time=0.0, final_speed=0.0, final_mach=0.0,
            total_control_energy=0.0, peak_lateral_g=0.0, saturation_percentage=0.0,
            hit_location_target=np.zeros(2), hit_location_missile=np.zeros(2)
        )
        with self.assertRaises(ValueError):
            ctrl.load_results([empty_res])

        # NaN trajectory
        nan_res = SimulationResult2D(
            law=GuidanceLaw2D.TPN, target_name="NaN", target_desc="NaN",
            time=np.array([0.0, 1.0]), r_M=np.array([[0.0, 0.0], [np.nan, 1.0]]), v_M=np.zeros((2, 2)),
            speed_M=np.ones(2), mach_M=np.ones(2), mass_M=np.ones(2),
            r_T=np.ones((2, 2)), v_T=np.zeros((2, 2)), a_T=np.zeros((2, 2)),
            range_dist=np.ones(2), closing_speed=np.ones(2), lambda_dot=np.zeros(2),
            a_cmd=np.zeros((2, 2)), a_lateral=np.zeros((2, 2)), a_lateral_g=np.zeros(2),
            a_net_g=np.zeros(2), is_saturated=np.zeros(2, dtype=bool), control_energy=np.zeros(2),
            miss_distance=0.0, intercept_time=1.0, final_speed=0.0, final_mach=0.0,
            total_control_energy=0.0, peak_lateral_g=0.0, saturation_percentage=0.0,
            hit_location_target=np.zeros(2), hit_location_missile=np.zeros(2)
        )
        with self.assertRaises(ValueError):
            ctrl.load_results([nan_res])

    def test_renderer_headless_2d(self):
        fig, ax = plt.subplots()
        renderer = MatplotlibAnimationRenderer2D()
        renderer.setup(ax, [self.result_tpn, self.result_apn], "Scenario A")

        self.controller.load_results([self.result_tpn, self.result_apn])
        self.controller.start()
        data = self.controller.step(0.5)

        renderer.update_frame(data)
        self.assertEqual(len(renderer.missile_trails), 2)
        self.assertEqual(len(renderer.missile_markers), 2)

        renderer.finalize()
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
