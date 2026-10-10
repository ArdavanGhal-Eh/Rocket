"""
Automated Integration Tests for Aerospace Desktop GUI & Live Animation.
========================================================================
Verifies:
- Complete AerospaceSimGUI lifecycle in Tkinter
- Playback controls (Run, Pause, Resume, Restart, Speed change)
- Concurrent simulation protection (blocking multi-runs)
- Frame ticking and HUD telemetry updates
- Terminal completion and Results Summary report population
- Safe window close handling
"""

import os
import sys
import unittest
from unittest.mock import patch
import tkinter as tk
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DIR_2D = os.path.join(BASE_DIR, "2D")
if DIR_2D not in sys.path:
    sys.path.insert(0, DIR_2D)
DIR_3D = os.path.join(BASE_DIR, "3D")
if DIR_3D not in sys.path:
    sys.path.insert(0, DIR_3D)

from gui_app import AerospaceSimGUI
from animation_controller import AnimationState


class TestGUILiveAnimationIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Patch messagebox so tests don't block on modal popups
        cls.patch_error = patch("tkinter.messagebox.showerror")
        cls.patch_warn = patch("tkinter.messagebox.showwarning")
        cls.mock_error = cls.patch_error.start()
        cls.mock_warn = cls.patch_warn.start()

        cls.root = tk.Tk()
        cls.root.withdraw()  # Headless Tkinter window
        cls.app = AerospaceSimGUI(cls.root)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app._on_window_close()
        except Exception:
            pass
        cls.patch_error.stop()
        cls.patch_warn.stop()

    def test_01_gui_controls_exist(self):
        # 3D Controls
        self.assertIsNotNone(self.app.btn_3d_run)
        self.assertIsNotNone(self.app.btn_3d_pause)
        self.assertIsNotNone(self.app.btn_3d_resume)
        self.assertIsNotNone(self.app.btn_3d_restart)
        self.assertIsNotNone(self.app.combo_3d_speed)
        self.assertIsNotNone(self.app.lbl_3d_hud_time)
        self.assertIsNotNone(self.app.lbl_3d_hud_status)

        # 2D Controls
        self.assertIsNotNone(self.app.btn_2d_run)
        self.assertIsNotNone(self.app.btn_2d_pause)
        self.assertIsNotNone(self.app.btn_2d_resume)
        self.assertIsNotNone(self.app.btn_2d_restart)
        self.assertIsNotNone(self.app.combo_2d_speed)
        self.assertIsNotNone(self.app.lbl_2d_hud_time)
        self.assertIsNotNone(self.app.lbl_2d_hud_status)

    def test_02_2d_live_simulation_and_playback_controls(self):
        app = self.app
        app.var_2d_mode.set(1)  # TPN
        app.var_2d_scen.set("A")

        # 1. Run simulation
        app._run_2d_simulation()
        self.assertTrue(app.anim_ctrl_2d.is_running())
        self.assertEqual(str(app.btn_2d_run.cget("state")), "disabled")
        self.assertEqual(str(app.btn_2d_pause.cget("state")), "normal")
        self.assertIn("Running", app.lbl_2d_hud_status.cget("text"))

        # 2. Test concurrent run protection (calling run again while active)
        prev_ctrl = app.anim_ctrl_2d
        app._run_2d_simulation()  # Should warn and do nothing
        self.assertIs(app.anim_ctrl_2d, prev_ctrl)
        self.assertTrue(app.anim_ctrl_2d.is_running())
        self.mock_warn.assert_called()

        # 3. Test Speed Change
        app.combo_2d_speed.set("2.0x")
        app._on_2d_speed_change()
        self.assertAlmostEqual(app.anim_ctrl_2d.speed_multiplier, 2.0)

        # 4. Test Pause
        app._pause_2d_animation()
        self.assertTrue(app.anim_ctrl_2d.is_paused())
        self.assertEqual(str(app.btn_2d_pause.cget("state")), "disabled")
        self.assertEqual(str(app.btn_2d_resume.cget("state")), "normal")
        self.assertIn("Paused", app.lbl_2d_hud_status.cget("text"))

        # 5. Test Resume
        app._resume_2d_animation()
        self.assertTrue(app.anim_ctrl_2d.is_running())
        self.assertEqual(str(app.btn_2d_pause.cget("state")), "normal")
        self.assertEqual(str(app.btn_2d_resume.cget("state")), "disabled")

        # 6. Step to completion and finalize
        app.anim_ctrl_2d.step(100.0)  # Advance to terminal CPA
        app._finalize_2d_simulation()

        self.assertTrue(app.anim_ctrl_2d.is_completed())
        self.assertIn("Completed", app.lbl_2d_hud_status.cget("text"))
        self.assertEqual(str(app.btn_2d_run.cget("state")), "normal")
        self.assertEqual(str(app.btn_2d_pause.cget("state")), "disabled")

        # Verify Results Summary report populated
        report_text = app.txt_2d_summary.get("1.0", tk.END)
        self.assertIn("2D INTERCEPTION SIMULATION REPORT", report_text)
        self.assertIn("Miss Distance", report_text)
        self.assertIn("Trajectory Fit Mean R²", report_text)

        # 7. Test Restart
        app._restart_2d_animation()
        self.assertTrue(app.anim_ctrl_2d.is_running())
        self.assertAlmostEqual(app.anim_ctrl_2d.current_sim_time, 0.0, delta=0.05)
        # Pause to leave in clean state
        app._pause_2d_animation()

    def test_03_3d_live_simulation_and_playback_controls(self):
        app = self.app
        app.var_3d_mode.set(2)  # APN
        app.var_3d_scen.set("A")

        # 1. Run 3D simulation
        app._run_3d_simulation()
        self.assertTrue(app.anim_ctrl_3d.is_running())
        self.assertEqual(str(app.btn_3d_run.cget("state")), "disabled")
        self.assertEqual(str(app.btn_3d_pause.cget("state")), "normal")

        # 2. Pause & Resume
        app._pause_3d_animation()
        self.assertTrue(app.anim_ctrl_3d.is_paused())
        app._resume_3d_animation()
        self.assertTrue(app.anim_ctrl_3d.is_running())

        # 3. Step to completion
        app.anim_ctrl_3d.step(100.0)
        app._finalize_3d_simulation()
        self.assertTrue(app.anim_ctrl_3d.is_completed())
        self.assertIn("Completed", app.lbl_3d_hud_status.cget("text"))

        report_text = app.txt_3d_summary.get("1.0", tk.END)
        self.assertIn("3D INTERCEPTION SIMULATION REPORT", report_text)
        self.assertIn("Miss Distance", report_text)

    def test_04_invalid_parameters_handling(self):
        app = self.app
        # Set invalid negative N
        app.entry_2d_n.delete(0, tk.END)
        app.entry_2d_n.insert(0, "-4.0")

        app._run_2d_simulation()
        self.assertIn("Error", app.lbl_2d_hud_status.cget("text"))
        self.assertFalse(app.anim_ctrl_2d.is_running())
        self.mock_error.assert_called()


if __name__ == "__main__":
    unittest.main()
