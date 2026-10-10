"""
====================================================================================================
Aerospace Missile Interception Simulation Suite | 2D & 3D (TPN vs APN vs Pure Pursuit)
High-Fidelity Desktop Engineering GUI Suite for Missile Guidance, Navigation & Control (GNC)
Supports Live Real-Time Trajectory Animation, Playback Controls, Telemetry HUD, & WebGL Dashboards
====================================================================================================
"""

import os
import sys
import time
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Ensure 3D and 2D packages are accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.join(BASE_DIR, "3D")
DIR_2D = os.path.join(BASE_DIR, "2D")

if DIR_3D not in sys.path:
    sys.path.insert(0, DIR_3D)
if DIR_2D not in sys.path:
    sys.path.insert(0, DIR_2D)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import 3D modules
from missile_sim.config import MissileConfig, TargetConfig, SimulationConfig
from missile_sim.guidance import GuidanceLaw
from missile_sim.target import create_target_maneuver
from missile_sim.simulator import SimulationEngine
from missile_sim.fitting import TrajectoryFitter
from missile_sim.visualizer import SimulationVisualizer
from missile_sim.monte_carlo import MonteCarloSimulator, MonteCarloConfig

# Import 2D modules
from missile_sim_2d.config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from missile_sim_2d.guidance import GuidanceLaw2D
from missile_sim_2d.target import create_target_maneuver_2d
from missile_sim_2d.simulator import SimulationEngine2D
from missile_sim_2d.fitting import TrajectoryFitter2D
from missile_sim_2d.visualizer import SimulationVisualizer2D
from missile_sim_2d.monte_carlo import MonteCarloSimulator2D, MonteCarloConfig2D

# Import decoupled Animation Engine
from animation_controller import (
    AnimationPlaybackController,
    AnimationState,
    PlaybackStateData,
    MatplotlibAnimationRenderer2D,
    MatplotlibAnimationRenderer3D,
)


class AerospaceSimGUI:
    """Main Desktop GUI Application with Live Animation for 2D & 3D Missile Simulation."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Aerospace GNC Simulation Suite | 2D & 3D Missile-Target Interception")
        self.root.geometry("1380x920")
        self.root.minsize(1120, 750)

        # 3D Animation state
        self.anim_ctrl_3d = AnimationPlaybackController(speed_multiplier=1.0)
        self.anim_renderer_3d = MatplotlibAnimationRenderer3D()
        self.anim_timer_3d = None
        self.last_tick_time_3d = 0.0
        self.results_3d = []
        self.ax_3d = None

        # 2D Animation state
        self.anim_ctrl_2d = AnimationPlaybackController(speed_multiplier=1.0)
        self.anim_renderer_2d = MatplotlibAnimationRenderer2D()
        self.anim_timer_2d = None
        self.last_tick_time_2d = 0.0
        self.results_2d = []
        self.ax_2d = None

        # Safe window closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Apply clean styling
        self._setup_styles()

        # Build main layout
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", font=("Segoe UI", 9))
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[14, 6])
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground="#1a365d")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#4a5568")

        # Buttons
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#2b6cb0")
        style.map("Action.TButton", background=[("active", "#2c5282"), ("disabled", "#a0aec0")])
        style.configure("Pause.TButton", font=("Segoe UI", 9, "bold"), foreground="#1a202c", background="#edf2f7")
        style.configure("Resume.TButton", font=("Segoe UI", 9, "bold"), foreground="#ffffff", background="#2f855a")
        style.map("Resume.TButton", background=[("active", "#22543d"), ("disabled", "#a0aec0")])
        style.configure("Restart.TButton", font=("Segoe UI", 9, "bold"), foreground="#1a202c", background="#feebc8")
        style.configure("Web.TButton", font=("Segoe UI", 9, "bold"), foreground="#ffffff", background="#276749")
        style.map("Web.TButton", background=[("active", "#22543d")])

        # HUD Cards
        style.configure("HUD.TFrame", background="#f7fafc", relief=tk.GROOVE)
        style.configure("HUDTitle.TLabel", font=("Segoe UI", 9, "bold"), foreground="#2d3748", background="#f7fafc")
        style.configure("HUDTime.TLabel", font=("Consolas", 12, "bold"), foreground="#2b6cb0", background="#f7fafc")
        style.configure("HUDData.TLabel", font=("Segoe UI", 8), foreground="#4a5568", background="#f7fafc")

    def _build_header(self):
        header_frame = ttk.Frame(self.root, padding="10 8 10 8")
        header_frame.pack(fill=tk.X)

        lbl_title = ttk.Label(
            header_frame,
            text="Aerospace Missile Guidance & Flight Dynamics Simulation Suite",
            style="Header.TLabel",
        )
        lbl_title.pack(anchor="w")

        lbl_subtitle = ttk.Label(
            header_frame,
            text="Real-Time 2D & 3D Live Interception Animation: TPN vs APN vs Pure Pursuit (Nose-to-Target) | RK4 Physics",
            style="SubHeader.TLabel",
        )
        lbl_subtitle.pack(anchor="w")

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Tab 1: 3D Simulation
        self.tab_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3d, text="  3D Space Simulation  ")
        self._init_3d_tab(self.tab_3d)

        # Tab 2: 2D Simulation
        self.tab_2d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2d, text="  2D Planar Simulation  ")
        self._init_2d_tab(self.tab_2d)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready. Configure parameters and click 'Run Simulation' to start live animation.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, padding=(8, 4), font=("Segoe UI", 8))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # =========================================================================
    # 3D TAB IMPLEMENTATION
    # =========================================================================
    def _init_3d_tab(self, parent):
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Control Panel (Left - Scrollable)
        left_container = ttk.Frame(pane, width=390)
        pane.add(left_container, weight=0)

        canvas_scroll = tk.Canvas(left_container, width=380, highlightthickness=0)
        v_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
        ctrl_frame = ttk.Frame(canvas_scroll, padding=6)

        ctrl_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        win_id = canvas_scroll.create_window((0, 0), window=ctrl_frame, anchor="nw")
        canvas_scroll.bind("<Configure>", lambda e: canvas_scroll.itemconfig(win_id, width=e.width))
        canvas_scroll.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Display Panel (Right)
        disp_frame = ttk.Frame(pane)
        pane.add(disp_frame, weight=1)

        # 1. Guidance Law Selection
        grp_mode = ttk.LabelFrame(ctrl_frame, text="Guidance Algorithm (Mode)", padding=6)
        grp_mode.pack(fill=tk.X, pady=3)

        self.var_3d_mode = tk.IntVar(value=4)
        ttk.Radiobutton(grp_mode, text="Mode 1: 3D True Proportional Navigation (TPN)", variable=self.var_3d_mode, value=1).pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_mode, text="Mode 2: 3D Augmented Proportional Navigation (APN)", variable=self.var_3d_mode, value=2).pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_mode, text="Mode 3: 3D Pure Pursuit (Nose-to-Target)", variable=self.var_3d_mode, value=3).pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_mode, text="Mode 4: 3-Way Benchmark Comparison (TPN vs APN vs PP) [Default]", variable=self.var_3d_mode, value=4).pack(anchor="w", pady=1)

        # 2. Target Maneuver Scenario
        grp_scen = ttk.LabelFrame(ctrl_frame, text="Target Maneuver or Equation", padding=6)
        grp_scen.pack(fill=tk.X, pady=3)

        self.var_3d_scen = tk.StringVar(value="D")
        ttk.Radiobutton(grp_scen, text="A: Straight Line (Non-maneuvering)", variable=self.var_3d_scen, value="A").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="B: 3D Barrel Roll / Spiral Climb (~7.6G)", variable=self.var_3d_scen, value="B").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="C: High-G Inclined Turn (7.5G)", variable=self.var_3d_scen, value="C").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="D: High-G 3D S-Turn / Break Turns (7-9G)", variable=self.var_3d_scen, value="D").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="LINE: 3D Linear Path (r(t) = r0 + v*t)", variable=self.var_3d_scen, value="LINE").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="PARABOLA: 3D Parabolic Path (r(t) = r0 + v0*t + 0.5*a*t²)", variable=self.var_3d_scen, value="PARABOLA").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="CUSTOM: Custom Math Functions (x(t), y(t), z(t))", variable=self.var_3d_scen, value="CUSTOM").pack(anchor="w", pady=1)

        # 3. Custom Equations / Parameters Frame
        grp_eq = ttk.LabelFrame(ctrl_frame, text="Custom Target Equations / Parameters", padding=6)
        grp_eq.pack(fill=tk.X, pady=3)

        row_eq_x = ttk.Frame(grp_eq)
        row_eq_x.pack(fill=tk.X, pady=1)
        ttk.Label(row_eq_x, text="x(t) or vx or ax:").pack(side=tk.LEFT)
        self.entry_3d_eq_x = ttk.Entry(row_eq_x, width=24)
        self.entry_3d_eq_x.insert(0, "6000 - 240*t")
        self.entry_3d_eq_x.pack(side=tk.RIGHT)

        row_eq_y = ttk.Frame(grp_eq)
        row_eq_y.pack(fill=tk.X, pady=1)
        ttk.Label(row_eq_y, text="y(t) or vy or ay:").pack(side=tk.LEFT)
        self.entry_3d_eq_y = ttk.Entry(row_eq_y, width=24)
        self.entry_3d_eq_y.insert(0, "2500 + 350*sin(0.5*t)")
        self.entry_3d_eq_y.pack(side=tk.RIGHT)

        row_eq_z = ttk.Frame(grp_eq)
        row_eq_z.pack(fill=tk.X, pady=1)
        ttk.Label(row_eq_z, text="z(t) or vz or az:").pack(side=tk.LEFT)
        self.entry_3d_eq_z = ttk.Entry(row_eq_z, width=24)
        self.entry_3d_eq_z.insert(0, "5000 + 200*cos(0.5*t)")
        self.entry_3d_eq_z.pack(side=tk.RIGHT)

        lbl_eq_hint = ttk.Label(
            grp_eq,
            text="Supported: sin, cos, tan, exp, log, sqrt, abs, pi and math operators",
            font=("Segoe UI", 7),
            foreground="#555",
        )
        lbl_eq_hint.pack(anchor="w", pady=2)

        # 4. Parameters
        grp_params = ttk.LabelFrame(ctrl_frame, text="Physical & Control Parameters", padding=6)
        grp_params.pack(fill=tk.X, pady=3)

        row_n = ttk.Frame(grp_params)
        row_n.pack(fill=tk.X, pady=2)
        ttk.Label(row_n, text="Navigation Ratio (N):").pack(side=tk.LEFT)
        self.entry_3d_n = ttk.Entry(row_n, width=8)
        self.entry_3d_n.insert(0, "4.0")
        self.entry_3d_n.pack(side=tk.RIGHT)

        row_g = ttk.Frame(grp_params)
        row_g.pack(fill=tk.X, pady=2)
        ttk.Label(row_g, text="Lateral G-Limit:").pack(side=tk.LEFT)
        self.entry_3d_g = ttk.Entry(row_g, width=8)
        self.entry_3d_g.insert(0, "35.0")
        self.entry_3d_g.pack(side=tk.RIGHT)

        row_t = ttk.Frame(grp_params)
        row_t.pack(fill=tk.X, pady=2)
        ttk.Label(row_t, text="Rocket Thrust (N):").pack(side=tk.LEFT)
        self.entry_3d_thrust = ttk.Entry(row_t, width=8)
        self.entry_3d_thrust.insert(0, "17500")
        self.entry_3d_thrust.pack(side=tk.RIGHT)

        row_burn = ttk.Frame(grp_params)
        row_burn.pack(fill=tk.X, pady=2)
        ttk.Label(row_burn, text="Burn Time (s):").pack(side=tk.LEFT)
        self.entry_3d_burn = ttk.Entry(row_burn, width=8)
        self.entry_3d_burn.insert(0, "4.0")
        self.entry_3d_burn.pack(side=tk.RIGHT)

        # 5. Playback Controls Section
        grp_playback = ttk.LabelFrame(ctrl_frame, text="Simulation & Animation Controls", padding=6)
        grp_playback.pack(fill=tk.X, pady=4)

        self.btn_3d_run = ttk.Button(grp_playback, text="🚀 Run Simulation", style="Action.TButton", command=self._run_3d_simulation)
        self.btn_3d_run.pack(fill=tk.X, pady=2)

        row_ctrls = ttk.Frame(grp_playback)
        row_ctrls.pack(fill=tk.X, pady=3)

        self.btn_3d_pause = ttk.Button(row_ctrls, text="⏸ Pause", command=self._pause_3d_animation, state="disabled")
        self.btn_3d_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        self.btn_3d_resume = ttk.Button(row_ctrls, text="▶ Resume", style="Resume.TButton", command=self._resume_3d_animation, state="disabled")
        self.btn_3d_resume.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        self.btn_3d_restart = ttk.Button(row_ctrls, text="🔄 Restart", command=self._restart_3d_animation, state="disabled")
        self.btn_3d_restart.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        row_speed = ttk.Frame(grp_playback)
        row_speed.pack(fill=tk.X, pady=2)
        ttk.Label(row_speed, text="Animation Speed:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.combo_3d_speed = ttk.Combobox(
            row_speed,
            values=["0.25x", "0.5x", "1.0x (Real-Time)", "2.0x", "5.0x"],
            state="readonly",
            width=16,
        )
        self.combo_3d_speed.set("1.0x (Real-Time)")
        self.combo_3d_speed.bind("<<ComboboxSelected>>", self._on_3d_speed_change)
        self.combo_3d_speed.pack(side=tk.RIGHT)

        btn_web = ttk.Button(grp_playback, text="🌐 Open Interactive 3D WebGL Dashboard", style="Web.TButton", command=self._open_3d_html)
        btn_web.pack(fill=tk.X, pady=2)

        btn_mc = ttk.Button(grp_playback, text="🎲 Run Monte Carlo Analysis (20 Runs)", command=self._run_3d_monte_carlo)
        btn_mc.pack(fill=tk.X, pady=2)

        # 6. Live HUD / Telemetry Box
        grp_hud = ttk.LabelFrame(ctrl_frame, text="Live Flight Telemetry (HUD)", padding=6)
        grp_hud.pack(fill=tk.X, pady=3)

        self.lbl_3d_hud_status = ttk.Label(grp_hud, text="● Simulation Status: Ready", font=("Segoe UI", 9, "bold"), foreground="#2b6cb0")
        self.lbl_3d_hud_status.pack(anchor="w", pady=1)

        self.lbl_3d_hud_time = ttk.Label(grp_hud, text="Simulation Time: 0.00 s", font=("Consolas", 11, "bold"), foreground="#1a202c")
        self.lbl_3d_hud_time.pack(anchor="w", pady=1)

        self.lbl_3d_hud_speed = ttk.Label(grp_hud, text="Missile Speed: --- m/s (Mach ---)", font=("Segoe UI", 8))
        self.lbl_3d_hud_speed.pack(anchor="w")

        self.lbl_3d_hud_range = ttk.Label(grp_hud, text="Range to Target: --- m | Closing Vc: --- m/s", font=("Segoe UI", 8))
        self.lbl_3d_hud_range.pack(anchor="w")

        self.lbl_3d_hud_accel = ttk.Label(grp_hud, text="Steering: --- G | Saturation: NO", font=("Segoe UI", 8))
        self.lbl_3d_hud_accel.pack(anchor="w")

        self.lbl_3d_hud_pos_m = ttk.Label(grp_hud, text="Missile Pos: [---, ---, ---] m", font=("Segoe UI", 8))
        self.lbl_3d_hud_pos_m.pack(anchor="w")

        self.lbl_3d_hud_pos_t = ttk.Label(grp_hud, text="Target Pos: [---, ---, ---] m", font=("Segoe UI", 8))
        self.lbl_3d_hud_pos_t.pack(anchor="w")

        # 7. Results Summary Box (Left bottom)
        grp_summary = ttk.LabelFrame(ctrl_frame, text="Performance & Results Report", padding=4)
        grp_summary.pack(fill=tk.BOTH, expand=True, pady=3)

        self.txt_3d_summary = tk.Text(grp_summary, height=13, wrap=tk.NONE, font=("Consolas", 8))
        scroll_y = ttk.Scrollbar(grp_summary, orient=tk.VERTICAL, command=self.txt_3d_summary.yview)
        scroll_x = ttk.Scrollbar(grp_summary, orient=tk.HORIZONTAL, command=self.txt_3d_summary.xview)
        self.txt_3d_summary.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_3d_summary.pack(fill=tk.BOTH, expand=True)

        # --- Display Panel setup (Right) ---
        disp_notebook = ttk.Notebook(disp_frame)
        disp_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab for 3D View
        tab_view3d = ttk.Frame(disp_notebook)
        disp_notebook.add(tab_view3d, text="3D Live Trajectory View")

        # Top banner for 3D View
        bar_top_3d = ttk.Frame(tab_view3d, padding=4)
        bar_top_3d.pack(fill=tk.X)
        self.lbl_3d_top_banner = ttk.Label(
            bar_top_3d,
            text="Status: Ready | Time: 0.00 s | Speed: 1.0x",
            font=("Segoe UI", 9, "bold"),
            foreground="#2b6cb0",
        )
        self.lbl_3d_top_banner.pack(side=tk.LEFT)

        # Tab for Telemetry
        tab_telemetry = ttk.Frame(disp_notebook)
        disp_notebook.add(tab_telemetry, text="Flight Telemetry Panels")

        # 3D Matplotlib Canvas
        self.fig_3d = plt.figure(figsize=(7, 6), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=tab_view3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, tab_view3d)
        self.toolbar_3d.update()

        # Telemetry Matplotlib Canvas
        self.fig_telem_3d = plt.figure(figsize=(7, 6), dpi=100)
        self.canvas_telem_3d = FigureCanvasTkAgg(self.fig_telem_3d, master=tab_telemetry)
        self.canvas_telem_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_telem_3d = NavigationToolbar2Tk(self.canvas_telem_3d, tab_telemetry)
        self.toolbar_telem_3d.update()

        self.last_3d_html_path = os.path.join(DIR_3D, "outputs", "intercept_3d_mode4_scenD.html")

    def _parse_speed_string(self, speed_str: str) -> float:
        cleaned = speed_str.split()[0].replace("x", "")
        try:
            return float(cleaned)
        except ValueError:
            return 1.0

    def _on_3d_speed_change(self, event=None):
        speed_val = self._parse_speed_string(self.combo_3d_speed.get())
        self.anim_ctrl_3d.set_speed(speed_val)
        banner_text = self.lbl_3d_top_banner.cget("text")
        if "Speed:" in banner_text:
            base_part = banner_text.rsplit("Speed:", 1)[0]
            self.lbl_3d_top_banner.config(text=f"{base_part}Speed: {speed_val}x")

    def _run_3d_simulation(self):
        """Execute 3D simulation, initialize real trajectory animation, and start playback."""
        if self.anim_ctrl_3d.is_running():
            messagebox.showwarning("Simulation Running", "A simulation animation is already active. Please pause or restart it.")
            return

        try:
            mode = self.var_3d_mode.get()
            scenario_id = self.var_3d_scen.get()
            nav_n = float(self.entry_3d_n.get())
            max_g = float(self.entry_3d_g.get())
            thrust = float(self.entry_3d_thrust.get())
            t_burn = float(self.entry_3d_burn.get())

            if nav_n <= 0 or max_g <= 0 or thrust < 0 or t_burn < 0:
                raise ValueError("Physical parameters (N, G-Limit, Thrust, Burn Time) must be positive.")

            self.status_var.set("Simulation Running...")
            self.lbl_3d_hud_status.config(text="● Simulation Status: Running...", foreground="#276749")
            self.lbl_3d_top_banner.config(text=f"Status: Simulation Running... | Time: 0.00 s | Speed: {self.anim_ctrl_3d.speed_multiplier}x", foreground="#276749")
            self.root.update_idletasks()

            # Build configs
            m_cfg = MissileConfig(
                nav_ratio=nav_n,
                g_limit_lateral=max_g,
                thrust_nominal=thrust,
                t_burn=t_burn,
            )
            t_cfg = TargetConfig()
            s_cfg = SimulationConfig()

            target_kwargs = {}
            if scenario_id == "LINE":
                try:
                    vx = float(self.entry_3d_eq_x.get())
                    vy = float(self.entry_3d_eq_y.get())
                    vz = float(self.entry_3d_eq_z.get())
                    target_kwargs = {"heading_dir": np.array([vx, vy, vz], dtype=np.float64)}
                except Exception as e:
                    raise ValueError(f"Invalid LINE equation inputs: {e}")
            elif scenario_id == "PARABOLA":
                try:
                    ax = float(self.entry_3d_eq_x.get())
                    ay = float(self.entry_3d_eq_y.get())
                    az = float(self.entry_3d_eq_z.get())
                    target_kwargs = {"a_const": np.array([ax, ay, az], dtype=np.float64)}
                except Exception as e:
                    raise ValueError(f"Invalid PARABOLA equation inputs: {e}")
            elif scenario_id == "CUSTOM":
                x_expr = self.entry_3d_eq_x.get().strip() or "6000 - 240*t"
                y_expr = self.entry_3d_eq_y.get().strip() or "2500 + 350*sin(0.5*t)"
                z_expr = self.entry_3d_eq_z.get().strip() or "5000 + 200*cos(0.5*t)"
                target_kwargs = {"x_expr": x_expr, "y_expr": y_expr, "z_expr": z_expr}

            target = create_target_maneuver(scenario_id, t_cfg, **target_kwargs)
            engine = SimulationEngine(m_cfg, t_cfg, s_cfg)

            laws = []
            if mode == 1:
                laws = [GuidanceLaw.TPN]
            elif mode == 2:
                laws = [GuidanceLaw.APN]
            elif mode == 3:
                laws = [GuidanceLaw.PP]
            else:
                laws = [GuidanceLaw.TPN, GuidanceLaw.APN, GuidanceLaw.PP]

            results = [engine.run(law, target) for law in laws]
            self.results_3d = results

            # Setup Animation Controller and Renderer
            self.anim_ctrl_3d.load_results(results)
            speed_val = self._parse_speed_string(self.combo_3d_speed.get())
            self.anim_ctrl_3d.set_speed(speed_val)
            self.anim_ctrl_3d.start()

            self.anim_renderer_3d.setup(self.ax_3d, results, target_name=results[0].target_name)
            self.canvas_3d.draw()

            # Update Button States
            self.btn_3d_run.config(state="disabled")
            self.btn_3d_pause.config(state="normal")
            self.btn_3d_resume.config(state="disabled")
            self.btn_3d_restart.config(state="normal")

            # Start Non-blocking loop via Tkinter after()
            if self.anim_timer_3d:
                self.root.after_cancel(self.anim_timer_3d)
            self.last_tick_time_3d = time.time()
            self._tick_3d_animation()

        except Exception as e:
            self.lbl_3d_hud_status.config(text="● Simulation Status: Error", foreground="#c53030")
            self.status_var.set(f"Simulation Error: {str(e)}")
            messagebox.showerror("Simulation Error", f"Failed to execute simulation:\n{str(e)}")

    def _tick_3d_animation(self):
        """Non-blocking animation step driven by Tkinter after()."""
        if not self.anim_ctrl_3d.is_running():
            return

        now = time.time()
        real_elapsed = max(0.001, now - self.last_tick_time_3d)
        self.last_tick_time_3d = now

        frame_data = self.anim_ctrl_3d.step(real_elapsed)
        self.anim_renderer_3d.update_frame(frame_data)
        self.canvas_3d.draw_idle()

        # Update HUD
        self._update_3d_hud(frame_data)

        if frame_data.is_terminal:
            self._finalize_3d_simulation()
        else:
            self.anim_timer_3d = self.root.after(30, self._tick_3d_animation)

    def _update_3d_hud(self, data: PlaybackStateData):
        """Update live flight telemetry cards and banners."""
        self.lbl_3d_hud_time.config(text=f"Simulation Time: {data.sim_time:.2f} s")
        speed_val = self.anim_ctrl_3d.speed_multiplier
        status_txt = "Simulation Running..." if data.state == AnimationState.RUNNING else (
            "Simulation Paused" if data.state == AnimationState.PAUSED else "Simulation Completed"
        )
        self.lbl_3d_top_banner.config(text=f"Status: {status_txt} | Time: {data.sim_time:.2f} s | Speed: {speed_val}x")

        if data.missile_states:
            m0 = data.missile_states[0]
            self.lbl_3d_hud_speed.config(text=f"Missile Speed: {m0.speed:.1f} m/s (Mach {m0.mach:.2f})")
            self.lbl_3d_hud_range.config(text=f"Range to Target: {m0.range_dist:.1f} m | Closing Vc: {m0.closing_speed:.1f} m/s")
            sat_str = "YES (35G Limit)" if m0.is_saturated else "NO"
            self.lbl_3d_hud_accel.config(text=f"Steering: {m0.lateral_g:.2f} G | Saturation: {sat_str}")
            self.lbl_3d_hud_pos_m.config(text=f"Missile Pos: [{m0.r_M[0]:.1f}, {m0.r_M[1]:.1f}, {m0.r_M[2]:.1f}] m")
            self.lbl_3d_hud_pos_t.config(text=f"Target Pos: [{data.r_T[0]:.1f}, {data.r_T[1]:.1f}, {data.r_T[2]:.1f}] m")

    def _finalize_3d_simulation(self):
        """Complete 3D animation, leave final frame intact, and populate Results Report."""
        if self.anim_timer_3d:
            self.root.after_cancel(self.anim_timer_3d)
            self.anim_timer_3d = None

        self.anim_renderer_3d.finalize()
        self.canvas_3d.draw_idle()

        self.lbl_3d_hud_status.config(text="● Simulation Status: Completed", foreground="#2b6cb0")
        self.status_var.set(f"Simulation Completed! Intercept at t = {self.results_3d[0].intercept_time:.2f}s (Miss = {self.results_3d[0].miss_distance:.3f}m).")
        self.lbl_3d_top_banner.config(
            text=f"Status: Simulation Completed | Final Time: {self.results_3d[0].intercept_time:.2f} s | Miss: {self.results_3d[0].miss_distance:.3f} m",
            foreground="#2b6cb0"
        )

        # Update button states
        self.btn_3d_run.config(state="normal")
        self.btn_3d_pause.config(state="disabled")
        self.btn_3d_resume.config(state="disabled")
        self.btn_3d_restart.config(state="normal")

        # Fit polynomials and populate Results Panel
        fitter = TrajectoryFitter(degree=6)
        fits = [fitter.fit(r.time, r.r_M) for r in self.results_3d]

        law_names = {
            GuidanceLaw.TPN: "3D TPN",
            GuidanceLaw.APN: "3D APN",
            GuidanceLaw.PP: "3D Pure Pursuit",
        }
        summary_lines = [
            "==========================================================================",
            f"            3D INTERCEPTION SIMULATION REPORT: {self.results_3d[0].target_name}",
            "==========================================================================",
        ]
        for res, fit in zip(self.results_3d, fits):
            law_str = law_names.get(res.law, getattr(res.law, "name", str(res.law)))
            summary_lines.append(f"\n[{law_str}]")
            summary_lines.append(f"  • Miss Distance (CPA): {res.miss_distance:.4f} m")
            summary_lines.append(f"  • Final Intercept Time: {res.intercept_time:.2f} s")
            summary_lines.append(f"  • Final Speed: {res.final_speed:.1f} m/s (Mach {res.final_mach:.2f})")
            summary_lines.append(f"  • Peak Lateral G: {res.peak_lateral_g:.2f} G (G-Saturation: {res.saturation_percentage:.2f}%)")
            summary_lines.append(f"  • Cumulative Control Energy: {res.total_control_energy:.2e} m²/s³")
            summary_lines.append(f"  • Final Missile Position: [{res.hit_location_missile[0]:.1f}, {res.hit_location_missile[1]:.1f}, {res.hit_location_missile[2]:.1f}] m")
            summary_lines.append(f"  • Final Target Position: [{res.hit_location_target[0]:.1f}, {res.hit_location_target[1]:.1f}, {res.hit_location_target[2]:.1f}] m")
            summary_lines.append(f"  • Trajectory Fit Mean R²: {fit.overall_mean_r2:.6f}")
            summary_lines.append(f"    - x(t) = {fit.x_fit.formula_str}")
            summary_lines.append(f"    - y(t) = {fit.y_fit.formula_str}")
            summary_lines.append(f"    - z(t) = {fit.z_fit.formula_str}")

        self.txt_3d_summary.delete("1.0", tk.END)
        self.txt_3d_summary.insert(tk.END, "\n".join(summary_lines))

        # Update Telemetry Canvas
        self._update_3d_telemetry_canvas()

        # Generate HTML output
        mode = self.var_3d_mode.get()
        scenario_id = self.var_3d_scen.get()
        viz = SimulationVisualizer()
        out_html = os.path.join(DIR_3D, "outputs", f"intercept_3d_gui_mode{mode}_scen{scenario_id}.html")
        viz.generate_interactive_plotly_3d(self.results_3d, output_html=out_html)
        self.last_3d_html_path = out_html

    def _update_3d_telemetry_canvas(self):
        """Update static analytical telemetry graphs on Telemetry tab."""
        self.fig_telem_3d.clf()
        axes = self.fig_telem_3d.subplots(2, 2)
        colors = {
            GuidanceLaw.TPN: "#1f77b4",
            GuidanceLaw.APN: "#2ca02c",
            GuidanceLaw.PP: "#d97706",
        }
        law_names = {
            GuidanceLaw.TPN: "3D TPN",
            GuidanceLaw.APN: "3D APN",
            GuidanceLaw.PP: "3D Pure Pursuit",
        }
        max_g = float(self.entry_3d_g.get())
        t_burn = float(self.entry_3d_burn.get())

        for res in self.results_3d:
            c = colors.get(res.law, "blue")
            lbl = law_names.get(res.law, getattr(res.law, "name", str(res.law)))
            axes[0, 0].plot(res.time, res.a_lateral_g, color=c, lw=1.8, label=f"{lbl} ({res.peak_lateral_g:.1f}G)")
            axes[0, 1].plot(res.time, res.speed_M, color=c, lw=1.8, label=f"{lbl} Speed")
            axes[1, 0].plot(res.time, res.range_dist, color=c, lw=1.8, label=f"{lbl} Range")
            axes[1, 1].plot(res.time, res.control_energy, color=c, lw=1.8, label=f"{lbl} Energy")

        axes[0, 0].axhline(max_g, color="crimson", linestyle="--", lw=1.5, label=f"Limit ({max_g}G)")
        axes[0, 0].set_title("Lateral Acceleration vs Limit (G)", fontsize=9, fontweight="bold")
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, linestyle=":", alpha=0.5)

        axes[0, 1].axvline(t_burn, color="orange", linestyle=":", lw=1.5, label="Burnout")
        axes[0, 1].set_title("Speed Profile (Boost / Coast)", fontsize=9, fontweight="bold")
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].grid(True, linestyle=":", alpha=0.5)

        axes[1, 0].set_title("Relative Range to Target (m)", fontsize=9, fontweight="bold")
        axes[1, 0].grid(True, linestyle=":", alpha=0.5)

        axes[1, 1].set_title("Control Energy Integral (m²/s³)", fontsize=9, fontweight="bold")
        axes[1, 1].grid(True, linestyle=":", alpha=0.5)

        self.fig_telem_3d.tight_layout()
        self.canvas_telem_3d.draw()

    def _pause_3d_animation(self):
        if self.anim_ctrl_3d.is_running():
            self.anim_ctrl_3d.pause()
            if self.anim_timer_3d:
                self.root.after_cancel(self.anim_timer_3d)
                self.anim_timer_3d = None
            self.lbl_3d_hud_status.config(text="● Simulation Status: Paused", foreground="#dd6b20")
            self.lbl_3d_top_banner.config(text=f"Status: Simulation Paused | Time: {self.anim_ctrl_3d.current_sim_time:.2f} s | Speed: {self.anim_ctrl_3d.speed_multiplier}x", foreground="#dd6b20")
            self.status_var.set("Simulation Paused")
            self.btn_3d_pause.config(state="disabled")
            self.btn_3d_resume.config(state="normal")

    def _resume_3d_animation(self):
        if self.anim_ctrl_3d.is_paused():
            self.anim_ctrl_3d.resume()
            self.last_tick_time_3d = time.time()
            self.lbl_3d_hud_status.config(text="● Simulation Status: Running...", foreground="#276749")
            self.lbl_3d_top_banner.config(text=f"Status: Simulation Running... | Time: {self.anim_ctrl_3d.current_sim_time:.2f} s | Speed: {self.anim_ctrl_3d.speed_multiplier}x", foreground="#276749")
            self.status_var.set("Simulation Running...")
            self.btn_3d_pause.config(state="normal")
            self.btn_3d_resume.config(state="disabled")
            self._tick_3d_animation()

    def _restart_3d_animation(self):
        if not self.results_3d:
            self._run_3d_simulation()
            return

        if self.anim_timer_3d:
            self.root.after_cancel(self.anim_timer_3d)
            self.anim_timer_3d = None

        self.anim_ctrl_3d.restart()
        self.anim_renderer_3d.setup(self.ax_3d, self.results_3d, target_name=self.results_3d[0].target_name)
        self.canvas_3d.draw()

        self.lbl_3d_hud_status.config(text="● Simulation Status: Running...", foreground="#276749")
        self.lbl_3d_top_banner.config(text=f"Status: Simulation Running... | Time: 0.00 s | Speed: {self.anim_ctrl_3d.speed_multiplier}x", foreground="#276749")
        self.status_var.set("Simulation Running...")

        self.btn_3d_run.config(state="disabled")
        self.btn_3d_pause.config(state="normal")
        self.btn_3d_resume.config(state="disabled")
        self.btn_3d_restart.config(state="normal")

        self.last_tick_time_3d = time.time()
        self._tick_3d_animation()

    def _open_3d_html(self):
        if os.path.exists(self.last_3d_html_path):
            webbrowser.open(f"file://{os.path.abspath(self.last_3d_html_path)}")
        else:
            fb = os.path.join(DIR_3D, "outputs", "intercept_3d_mode4_scenD.html")
            if os.path.exists(fb):
                webbrowser.open(f"file://{os.path.abspath(fb)}")
            else:
                messagebox.showinfo("Notice", "Please click 'Run Simulation' first to generate the web dashboard.")

    def _run_3d_monte_carlo(self):
        """Execute 3D Monte Carlo dispersion analysis from GUI."""
        try:
            self.status_var.set("Running 3D Monte Carlo dispersion analysis (20 runs)...")
            self.root.update_idletasks()

            mode = self.var_3d_mode.get()
            scen = self.var_3d_scen.get()
            target_kwargs = {}
            if scen == "LINE":
                target_kwargs = {"speed": 300.0}
            elif scen == "PARABOLA":
                target_kwargs = {"a_const": np.array([0.0, 15.0, -9.81])}
            elif scen == "CUSTOM":
                target_kwargs = {
                    "x_expr": self.entry_3d_eq_x.get().strip() or "6000 - 240*t",
                    "y_expr": self.entry_3d_eq_y.get().strip() or "2500 + 350*sin(0.5*t)",
                    "z_expr": self.entry_3d_eq_z.get().strip() or "5000 + 200*cos(0.5*t)",
                }

            laws = [GuidanceLaw.TPN, GuidanceLaw.APN, GuidanceLaw.PP] if mode == 4 else (
                [GuidanceLaw.TPN] if mode == 1 else ([GuidanceLaw.APN] if mode == 2 else [GuidanceLaw.PP])
            )

            mc_sim = MonteCarloSimulator()
            mc_cfg = MonteCarloConfig(num_runs=20)

            report_sections = [
                "==========================================================================",
                f"       3D MONTE CARLO DISPERSION CAMPAIGN (20 RUNS) | Scenario: {scen}",
                "==========================================================================",
            ]
            for law in laws:
                summary = mc_sim.run_campaign(
                    guidance_law=law,
                    scenario_id=scen,
                    target_kwargs=target_kwargs,
                    mc_config=mc_cfg,
                )
                report_sections.append(summary.summary_table())

            self.txt_3d_summary.delete("1.0", tk.END)
            self.txt_3d_summary.insert(tk.END, "\n\n".join(report_sections))
            self.status_var.set("3D Monte Carlo campaign complete!")
        except Exception as e:
            messagebox.showerror("Monte Carlo Error", f"Error during Monte Carlo analysis:\n{str(e)}")
            self.status_var.set("Error during 3D Monte Carlo run.")

    # =========================================================================
    # 2D TAB IMPLEMENTATION
    # =========================================================================
    def _init_2d_tab(self, parent):
        pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Control Panel (Left - Scrollable)
        left_container = ttk.Frame(pane, width=390)
        pane.add(left_container, weight=0)

        canvas_scroll = tk.Canvas(left_container, width=380, highlightthickness=0)
        v_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
        ctrl_frame = ttk.Frame(canvas_scroll, padding=6)

        ctrl_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        win_id = canvas_scroll.create_window((0, 0), window=ctrl_frame, anchor="nw")
        canvas_scroll.bind("<Configure>", lambda e: canvas_scroll.itemconfig(win_id, width=e.width))
        canvas_scroll.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Display Panel (Right)
        disp_frame = ttk.Frame(pane)
        pane.add(disp_frame, weight=1)

        # 1. Guidance Law
        grp_mode = ttk.LabelFrame(ctrl_frame, text="Guidance Algorithm (Mode)", padding=6)
        grp_mode.pack(fill=tk.X, pady=3)

        self.var_2d_mode = tk.IntVar(value=4)
        ttk.Radiobutton(grp_mode, text="Mode 1: 2D True Proportional Navigation (TPN)", variable=self.var_2d_mode, value=1).pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_mode, text="Mode 2: 2D Augmented Proportional Navigation (APN)", variable=self.var_2d_mode, value=2).pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_mode, text="Mode 3: 2D Pure Pursuit (Nose-to-Target)", variable=self.var_2d_mode, value=3).pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_mode, text="Mode 4: 3-Way Benchmark Comparison (TPN vs APN vs PP) [Default]", variable=self.var_2d_mode, value=4).pack(anchor="w", pady=1)

        # 2. Scenario
        grp_scen = ttk.LabelFrame(ctrl_frame, text="Planar Target Maneuver / Equation", padding=6)
        grp_scen.pack(fill=tk.X, pady=3)

        self.var_2d_scen = tk.StringVar(value="D")
        ttk.Radiobutton(grp_scen, text="A: Straight Line (Non-maneuvering)", variable=self.var_2d_scen, value="A").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="B: Sinusoidal Weave (~7.5G)", variable=self.var_2d_scen, value="B").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="C: Circular Turn (7.5G)", variable=self.var_2d_scen, value="C").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="D: High-G 2D S-Turn (7-9G)", variable=self.var_2d_scen, value="D").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="LINE: Linear Equation (y = m*x + c)", variable=self.var_2d_scen, value="LINE").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="PARABOLA: Parabolic Equation (y = a*x² + b*x + c)", variable=self.var_2d_scen, value="PARABOLA").pack(anchor="w", pady=1)
        ttk.Radiobutton(grp_scen, text="CUSTOM: Custom Math Functions (x(t), y(t))", variable=self.var_2d_scen, value="CUSTOM").pack(anchor="w", pady=1)

        # 3. Custom Equations / Parameters Frame
        grp_eq_2d = ttk.LabelFrame(ctrl_frame, text="Custom Target Equations / Parameters", padding=6)
        grp_eq_2d.pack(fill=tk.X, pady=3)

        row_eq2_x = ttk.Frame(grp_eq_2d)
        row_eq2_x.pack(fill=tk.X, pady=1)
        ttk.Label(row_eq2_x, text="x(t) or Slope m or Param a:").pack(side=tk.LEFT)
        self.entry_2d_eq_x = ttk.Entry(row_eq2_x, width=22)
        self.entry_2d_eq_x.insert(0, "6000 - 250*t")
        self.entry_2d_eq_x.pack(side=tk.RIGHT)

        row_eq2_y = ttk.Frame(grp_eq_2d)
        row_eq2_y.pack(fill=tk.X, pady=1)
        ttk.Label(row_eq2_y, text="y(t) or Intercept c or Param b:").pack(side=tk.LEFT)
        self.entry_2d_eq_y = ttk.Entry(row_eq2_y, width=22)
        self.entry_2d_eq_y.insert(0, "2500 + 400*sin(0.6*t)")
        self.entry_2d_eq_y.pack(side=tk.RIGHT)

        row_eq2_c = ttk.Frame(grp_eq_2d)
        row_eq2_c.pack(fill=tk.X, pady=1)
        ttk.Label(row_eq2_c, text="Parabola c Constant:").pack(side=tk.LEFT)
        self.entry_2d_eq_c = ttk.Entry(row_eq2_c, width=22)
        self.entry_2d_eq_c.insert(0, "1900.0")
        self.entry_2d_eq_c.pack(side=tk.RIGHT)

        lbl_eq2_hint = ttk.Label(
            grp_eq_2d,
            text="Formulas with parameter 't' (e.g. 100*t, 2500 + 500*sin(0.5*t))",
            font=("Segoe UI", 7),
            foreground="#555",
        )
        lbl_eq2_hint.pack(anchor="w", pady=2)

        # 4. Parameters
        grp_params = ttk.LabelFrame(ctrl_frame, text="Flight & Guidance Parameters", padding=6)
        grp_params.pack(fill=tk.X, pady=3)

        row_n = ttk.Frame(grp_params)
        row_n.pack(fill=tk.X, pady=2)
        ttk.Label(row_n, text="Navigation Ratio (N):").pack(side=tk.LEFT)
        self.entry_2d_n = ttk.Entry(row_n, width=8)
        self.entry_2d_n.insert(0, "4.0")
        self.entry_2d_n.pack(side=tk.RIGHT)

        row_g = ttk.Frame(grp_params)
        row_g.pack(fill=tk.X, pady=2)
        ttk.Label(row_g, text="Lateral G-Limit:").pack(side=tk.LEFT)
        self.entry_2d_g = ttk.Entry(row_g, width=8)
        self.entry_2d_g.insert(0, "35.0")
        self.entry_2d_g.pack(side=tk.RIGHT)

        # 5. Playback Controls Section
        grp_playback = ttk.LabelFrame(ctrl_frame, text="Simulation & Animation Controls", padding=6)
        grp_playback.pack(fill=tk.X, pady=4)

        self.btn_2d_run = ttk.Button(grp_playback, text="🚀 Run Simulation", style="Action.TButton", command=self._run_2d_simulation)
        self.btn_2d_run.pack(fill=tk.X, pady=2)

        row_ctrls_2d = ttk.Frame(grp_playback)
        row_ctrls_2d.pack(fill=tk.X, pady=3)

        self.btn_2d_pause = ttk.Button(row_ctrls_2d, text="⏸ Pause", command=self._pause_2d_animation, state="disabled")
        self.btn_2d_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        self.btn_2d_resume = ttk.Button(row_ctrls_2d, text="▶ Resume", style="Resume.TButton", command=self._resume_2d_animation, state="disabled")
        self.btn_2d_resume.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        self.btn_2d_restart = ttk.Button(row_ctrls_2d, text="🔄 Restart", command=self._restart_2d_animation, state="disabled")
        self.btn_2d_restart.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        row_speed_2d = ttk.Frame(grp_playback)
        row_speed_2d.pack(fill=tk.X, pady=2)
        ttk.Label(row_speed_2d, text="Animation Speed:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.combo_2d_speed = ttk.Combobox(
            row_speed_2d,
            values=["0.25x", "0.5x", "1.0x (Real-Time)", "2.0x", "5.0x"],
            state="readonly",
            width=16,
        )
        self.combo_2d_speed.set("1.0x (Real-Time)")
        self.combo_2d_speed.bind("<<ComboboxSelected>>", self._on_2d_speed_change)
        self.combo_2d_speed.pack(side=tk.RIGHT)

        btn_web = ttk.Button(grp_playback, text="🌐 Open Interactive 2D Web Dashboard", style="Web.TButton", command=self._open_2d_html)
        btn_web.pack(fill=tk.X, pady=2)

        btn_mc_2d = ttk.Button(grp_playback, text="🎲 Run Monte Carlo Analysis (20 Runs)", command=self._run_2d_monte_carlo)
        btn_mc_2d.pack(fill=tk.X, pady=2)

        # 6. Live HUD / Telemetry Box
        grp_hud = ttk.LabelFrame(ctrl_frame, text="Live Flight Telemetry (HUD)", padding=6)
        grp_hud.pack(fill=tk.X, pady=3)

        self.lbl_2d_hud_status = ttk.Label(grp_hud, text="● Simulation Status: Ready", font=("Segoe UI", 9, "bold"), foreground="#2b6cb0")
        self.lbl_2d_hud_status.pack(anchor="w", pady=1)

        self.lbl_2d_hud_time = ttk.Label(grp_hud, text="Simulation Time: 0.00 s", font=("Consolas", 11, "bold"), foreground="#1a202c")
        self.lbl_2d_hud_time.pack(anchor="w", pady=1)

        self.lbl_2d_hud_speed = ttk.Label(grp_hud, text="Missile Speed: --- m/s (Mach ---)", font=("Segoe UI", 8))
        self.lbl_2d_hud_speed.pack(anchor="w")

        self.lbl_2d_hud_range = ttk.Label(grp_hud, text="Range to Target: --- m | Closing Vc: --- m/s", font=("Segoe UI", 8))
        self.lbl_2d_hud_range.pack(anchor="w")

        self.lbl_2d_hud_accel = ttk.Label(grp_hud, text="Steering: --- G | Saturation: NO", font=("Segoe UI", 8))
        self.lbl_2d_hud_accel.pack(anchor="w")

        self.lbl_2d_hud_pos_m = ttk.Label(grp_hud, text="Missile Pos: [---, ---] m", font=("Segoe UI", 8))
        self.lbl_2d_hud_pos_m.pack(anchor="w")

        self.lbl_2d_hud_pos_t = ttk.Label(grp_hud, text="Target Pos: [---, ---] m", font=("Segoe UI", 8))
        self.lbl_2d_hud_pos_t.pack(anchor="w")

        # 7. Results Summary Box
        grp_summary = ttk.LabelFrame(ctrl_frame, text="Performance & Results Report", padding=4)
        grp_summary.pack(fill=tk.BOTH, expand=True, pady=3)

        self.txt_2d_summary = tk.Text(grp_summary, height=13, wrap=tk.NONE, font=("Consolas", 8))
        scroll_y = ttk.Scrollbar(grp_summary, orient=tk.VERTICAL, command=self.txt_2d_summary.yview)
        scroll_x = ttk.Scrollbar(grp_summary, orient=tk.HORIZONTAL, command=self.txt_2d_summary.xview)
        self.txt_2d_summary.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_2d_summary.pack(fill=tk.BOTH, expand=True)

        # Display Panel
        disp_notebook = ttk.Notebook(disp_frame)
        disp_notebook.pack(fill=tk.BOTH, expand=True)

        tab_view2d = ttk.Frame(disp_notebook)
        disp_notebook.add(tab_view2d, text="Planar 2D Live Trajectory View")

        bar_top_2d = ttk.Frame(tab_view2d, padding=4)
        bar_top_2d.pack(fill=tk.X)
        self.lbl_2d_top_banner = ttk.Label(
            bar_top_2d,
            text="Status: Ready | Time: 0.00 s | Speed: 1.0x",
            font=("Segoe UI", 9, "bold"),
            foreground="#2b6cb0",
        )
        self.lbl_2d_top_banner.pack(side=tk.LEFT)

        tab_telem2d = ttk.Frame(disp_notebook)
        disp_notebook.add(tab_telem2d, text="Flight Telemetry Panels")

        self.fig_2d = plt.figure(figsize=(7, 6), dpi=100)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=tab_view2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_2d = NavigationToolbar2Tk(self.canvas_2d, tab_view2d)
        self.toolbar_2d.update()

        self.fig_telem_2d = plt.figure(figsize=(7, 6), dpi=100)
        self.canvas_telem_2d = FigureCanvasTkAgg(self.fig_telem_2d, master=tab_telem2d)
        self.canvas_telem_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_telem_2d = NavigationToolbar2Tk(self.canvas_telem_2d, tab_telem2d)
        self.toolbar_telem_2d.update()

        self.last_2d_html_path = os.path.join(DIR_2D, "outputs", "intercept_2d_mode4_scenD.html")

    def _on_2d_speed_change(self, event=None):
        speed_val = self._parse_speed_string(self.combo_2d_speed.get())
        self.anim_ctrl_2d.set_speed(speed_val)
        banner_text = self.lbl_2d_top_banner.cget("text")
        if "Speed:" in banner_text:
            base_part = banner_text.rsplit("Speed:", 1)[0]
            self.lbl_2d_top_banner.config(text=f"{base_part}Speed: {speed_val}x")

    def _run_2d_simulation(self):
        """Execute 2D planar simulation, setup animation, and begin playback."""
        if self.anim_ctrl_2d.is_running():
            messagebox.showwarning("Simulation Running", "A simulation animation is already active. Please pause or restart it.")
            return

        try:
            mode = self.var_2d_mode.get()
            scenario_id = self.var_2d_scen.get()
            nav_n = float(self.entry_2d_n.get())
            max_g = float(self.entry_2d_g.get())

            if nav_n <= 0 or max_g <= 0:
                raise ValueError("Navigation ratio (N) and lateral G-limit must be positive.")

            self.status_var.set("Simulation Running...")
            self.lbl_2d_hud_status.config(text="● Simulation Status: Running...", foreground="#276749")
            self.lbl_2d_top_banner.config(text=f"Status: Simulation Running... | Time: 0.00 s | Speed: {self.anim_ctrl_2d.speed_multiplier}x", foreground="#276749")
            self.root.update_idletasks()

            m_cfg = MissileConfig2D(nav_ratio=nav_n, g_limit_lateral=max_g)
            t_cfg = TargetConfig2D()
            s_cfg = SimulationConfig2D()

            target_kwargs = {}
            if scenario_id == "LINE":
                try:
                    slope = float(self.entry_2d_eq_x.get())
                    intercept = float(self.entry_2d_eq_y.get())
                    target_kwargs = {"slope": slope, "intercept": intercept}
                except Exception as e:
                    raise ValueError(f"Invalid LINE equation inputs: {e}")
            elif scenario_id == "PARABOLA":
                try:
                    a = float(self.entry_2d_eq_x.get())
                    b = float(self.entry_2d_eq_y.get())
                    c = float(self.entry_2d_eq_c.get())
                    target_kwargs = {"a": a, "b": b, "c": c}
                except Exception as e:
                    raise ValueError(f"Invalid PARABOLA equation inputs: {e}")
            elif scenario_id == "CUSTOM":
                x_expr = self.entry_2d_eq_x.get().strip() or "6000 - 250*t"
                y_expr = self.entry_2d_eq_y.get().strip() or "2500 + 400*sin(0.6*t)"
                target_kwargs = {"x_expr": x_expr, "y_expr": y_expr}

            target = create_target_maneuver_2d(scenario_id, t_cfg, **target_kwargs)
            engine = SimulationEngine2D(m_cfg, t_cfg, s_cfg)

            laws = []
            if mode == 1:
                laws = [GuidanceLaw2D.TPN]
            elif mode == 2:
                laws = [GuidanceLaw2D.APN]
            elif mode == 3:
                laws = [GuidanceLaw2D.PP]
            else:
                laws = [GuidanceLaw2D.TPN, GuidanceLaw2D.APN, GuidanceLaw2D.PP]

            results = [engine.run(law, target) for law in laws]
            self.results_2d = results

            # Setup Animation Controller and Renderer
            self.anim_ctrl_2d.load_results(results)
            speed_val = self._parse_speed_string(self.combo_2d_speed.get())
            self.anim_ctrl_2d.set_speed(speed_val)
            self.anim_ctrl_2d.start()

            self.anim_renderer_2d.setup(self.ax_2d, results, target_name=results[0].target_name)
            self.canvas_2d.draw()

            # Update Button States
            self.btn_2d_run.config(state="disabled")
            self.btn_2d_pause.config(state="normal")
            self.btn_2d_resume.config(state="disabled")
            self.btn_2d_restart.config(state="normal")

            if self.anim_timer_2d:
                self.root.after_cancel(self.anim_timer_2d)
            self.last_tick_time_2d = time.time()
            self._tick_2d_animation()

        except Exception as e:
            self.lbl_2d_hud_status.config(text="● Simulation Status: Error", foreground="#c53030")
            self.status_var.set(f"Simulation Error: {str(e)}")
            messagebox.showerror("Simulation Error", f"Failed to execute 2D simulation:\n{str(e)}")

    def _tick_2d_animation(self):
        """Non-blocking 2D animation step driven by Tkinter after()."""
        if not self.anim_ctrl_2d.is_running():
            return

        now = time.time()
        real_elapsed = max(0.001, now - self.last_tick_time_2d)
        self.last_tick_time_2d = now

        frame_data = self.anim_ctrl_2d.step(real_elapsed)
        self.anim_renderer_2d.update_frame(frame_data)
        self.canvas_2d.draw_idle()

        # Update HUD
        self._update_2d_hud(frame_data)

        if frame_data.is_terminal:
            self._finalize_2d_simulation()
        else:
            self.anim_timer_2d = self.root.after(30, self._tick_2d_animation)

    def _update_2d_hud(self, data: PlaybackStateData):
        """Update live planar 2D flight telemetry cards and banners."""
        self.lbl_2d_hud_time.config(text=f"Simulation Time: {data.sim_time:.2f} s")
        speed_val = self.anim_ctrl_2d.speed_multiplier
        status_txt = "Simulation Running..." if data.state == AnimationState.RUNNING else (
            "Simulation Paused" if data.state == AnimationState.PAUSED else "Simulation Completed"
        )
        self.lbl_2d_top_banner.config(text=f"Status: {status_txt} | Time: {data.sim_time:.2f} s | Speed: {speed_val}x")

        if data.missile_states:
            m0 = data.missile_states[0]
            self.lbl_2d_hud_speed.config(text=f"Missile Speed: {m0.speed:.1f} m/s (Mach {m0.mach:.2f})")
            self.lbl_2d_hud_range.config(text=f"Range to Target: {m0.range_dist:.1f} m | Closing Vc: {m0.closing_speed:.1f} m/s")
            sat_str = "YES (35G Limit)" if m0.is_saturated else "NO"
            self.lbl_2d_hud_accel.config(text=f"Steering: {m0.lateral_g:.2f} G | Saturation: {sat_str}")
            self.lbl_2d_hud_pos_m.config(text=f"Missile Pos: [{m0.r_M[0]:.1f}, {m0.r_M[1]:.1f}] m")
            self.lbl_2d_hud_pos_t.config(text=f"Target Pos: [{data.r_T[0]:.1f}, {data.r_T[1]:.1f}] m")

    def _finalize_2d_simulation(self):
        """Complete 2D animation, preserve last frame, and render final Performance Report."""
        if self.anim_timer_2d:
            self.root.after_cancel(self.anim_timer_2d)
            self.anim_timer_2d = None

        self.anim_renderer_2d.finalize()
        self.canvas_2d.draw_idle()

        self.lbl_2d_hud_status.config(text="● Simulation Status: Completed", foreground="#2b6cb0")
        self.status_var.set(f"Simulation Completed! Intercept at t = {self.results_2d[0].intercept_time:.2f}s (Miss = {self.results_2d[0].miss_distance:.3f}m).")
        self.lbl_2d_top_banner.config(
            text=f"Status: Simulation Completed | Final Time: {self.results_2d[0].intercept_time:.2f} s | Miss: {self.results_2d[0].miss_distance:.3f} m",
            foreground="#2b6cb0"
        )

        # Update button states
        self.btn_2d_run.config(state="normal")
        self.btn_2d_pause.config(state="disabled")
        self.btn_2d_resume.config(state="disabled")
        self.btn_2d_restart.config(state="normal")

        # Fit polynomials and populate Results Panel
        fitter = TrajectoryFitter2D(degree=6)
        fits = [fitter.fit(r.time, r.r_M) for r in self.results_2d]

        law_names = {
            GuidanceLaw2D.TPN: "2D TPN",
            GuidanceLaw2D.APN: "2D APN",
            GuidanceLaw2D.PP: "2D Pure Pursuit",
        }
        summary_lines = [
            "==========================================================================",
            f"            2D INTERCEPTION SIMULATION REPORT: {self.results_2d[0].target_name}",
            "==========================================================================",
        ]
        for res, fit in zip(self.results_2d, fits):
            law_str = law_names.get(res.law, getattr(res.law, "name", str(res.law)))
            summary_lines.append(f"\n[{law_str}]")
            summary_lines.append(f"  • Miss Distance (CPA): {res.miss_distance:.4f} m")
            summary_lines.append(f"  • Final Intercept Time: {res.intercept_time:.2f} s")
            summary_lines.append(f"  • Final Speed: {res.final_speed:.1f} m/s (Mach {res.final_mach:.2f})")
            summary_lines.append(f"  • Peak Lateral G: {res.peak_lateral_g:.2f} G (G-Saturation: {res.saturation_percentage:.2f}%)")
            summary_lines.append(f"  • Cumulative Control Energy: {res.total_control_energy:.2e} m²/s³")
            summary_lines.append(f"  • Final Missile Position: [{res.hit_location_missile[0]:.1f}, {res.hit_location_missile[1]:.1f}] m")
            summary_lines.append(f"  • Final Target Position: [{res.hit_location_target[0]:.1f}, {res.hit_location_target[1]:.1f}] m")
            summary_lines.append(f"  • Trajectory Fit Mean R²: {fit.overall_mean_r2:.6f}")
            summary_lines.append(f"    - x(t) = {fit.x_fit.formula_str}")
            summary_lines.append(f"    - y(t) = {fit.y_fit.formula_str}")

        self.txt_2d_summary.delete("1.0", tk.END)
        self.txt_2d_summary.insert(tk.END, "\n".join(summary_lines))

        # Update Telemetry Canvas
        self._update_2d_telemetry_canvas()

        mode = self.var_2d_mode.get()
        scenario_id = self.var_2d_scen.get()
        viz = SimulationVisualizer2D()
        out_html = os.path.join(DIR_2D, "outputs", f"intercept_2d_gui_mode{mode}_scen{scenario_id}.html")
        viz.generate_interactive_plotly_2d(self.results_2d, output_html=out_html)
        self.last_2d_html_path = out_html

    def _update_2d_telemetry_canvas(self):
        """Update static analytical telemetry graphs on Telemetry tab."""
        self.fig_telem_2d.clf()
        axes = self.fig_telem_2d.subplots(2, 2)
        colors = {
            GuidanceLaw2D.TPN: "#1f77b4",
            GuidanceLaw2D.APN: "#2ca02c",
            GuidanceLaw2D.PP: "#d97706",
        }
        law_names = {
            GuidanceLaw2D.TPN: "2D TPN",
            GuidanceLaw2D.APN: "2D APN",
            GuidanceLaw2D.PP: "2D Pure Pursuit",
        }
        max_g = float(self.entry_2d_g.get())

        for res in self.results_2d:
            c = colors.get(res.law, "blue")
            lbl = law_names.get(res.law, getattr(res.law, "name", str(res.law)))
            axes[0, 0].plot(res.time, res.a_lateral_g, color=c, lw=1.8, label=f"{lbl} ({res.peak_lateral_g:.1f}G)")
            axes[0, 1].plot(res.time, res.speed_M, color=c, lw=1.8, label=f"{lbl} Speed")
            axes[1, 0].plot(res.time, res.range_dist, color=c, lw=1.8, label=f"{lbl} Range")
            axes[1, 1].plot(res.time, res.control_energy, color=c, lw=1.8, label=f"{lbl} Energy")

        axes[0, 0].axhline(max_g, color="crimson", linestyle="--", lw=1.5, label=f"Limit ({max_g}G)")
        axes[0, 0].set_title("Lateral Steering Acceleration vs Limit (G)", fontsize=9, fontweight="bold")
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(True, linestyle=":", alpha=0.5)

        axes[0, 1].axvline(4.0, color="orange", linestyle=":", lw=1.5, label="Burnout")
        axes[0, 1].set_title("Speed Profile (Boost / Coast)", fontsize=9, fontweight="bold")
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].grid(True, linestyle=":", alpha=0.5)

        axes[1, 0].set_title("Relative Range to Target (m)", fontsize=9, fontweight="bold")
        axes[1, 0].grid(True, linestyle=":", alpha=0.5)

        axes[1, 1].set_title("Control Energy Integral (m²/s³)", fontsize=9, fontweight="bold")
        axes[1, 1].grid(True, linestyle=":", alpha=0.5)

        self.fig_telem_2d.tight_layout()
        self.canvas_telem_2d.draw()

    def _pause_2d_animation(self):
        if self.anim_ctrl_2d.is_running():
            self.anim_ctrl_2d.pause()
            if self.anim_timer_2d:
                self.root.after_cancel(self.anim_timer_2d)
                self.anim_timer_2d = None
            self.lbl_2d_hud_status.config(text="● Simulation Status: Paused", foreground="#dd6b20")
            self.lbl_2d_top_banner.config(text=f"Status: Simulation Paused | Time: {self.anim_ctrl_2d.current_sim_time:.2f} s | Speed: {self.anim_ctrl_2d.speed_multiplier}x", foreground="#dd6b20")
            self.status_var.set("Simulation Paused")
            self.btn_2d_pause.config(state="disabled")
            self.btn_2d_resume.config(state="normal")

    def _resume_2d_animation(self):
        if self.anim_ctrl_2d.is_paused():
            self.anim_ctrl_2d.resume()
            self.last_tick_time_2d = time.time()
            self.lbl_2d_hud_status.config(text="● Simulation Status: Running...", foreground="#276749")
            self.lbl_2d_top_banner.config(text=f"Status: Simulation Running... | Time: {self.anim_ctrl_2d.current_sim_time:.2f} s | Speed: {self.anim_ctrl_2d.speed_multiplier}x", foreground="#276749")
            self.status_var.set("Simulation Running...")
            self.btn_2d_pause.config(state="normal")
            self.btn_2d_resume.config(state="disabled")
            self._tick_2d_animation()

    def _restart_2d_animation(self):
        if not self.results_2d:
            self._run_2d_simulation()
            return

        if self.anim_timer_2d:
            self.root.after_cancel(self.anim_timer_2d)
            self.anim_timer_2d = None

        self.anim_ctrl_2d.restart()
        self.anim_renderer_2d.setup(self.ax_2d, self.results_2d, target_name=self.results_2d[0].target_name)
        self.canvas_2d.draw()

        self.lbl_2d_hud_status.config(text="● Simulation Status: Running...", foreground="#276749")
        self.lbl_2d_top_banner.config(text=f"Status: Simulation Running... | Time: 0.00 s | Speed: {self.anim_ctrl_2d.speed_multiplier}x", foreground="#276749")
        self.status_var.set("Simulation Running...")

        self.btn_2d_run.config(state="disabled")
        self.btn_2d_pause.config(state="normal")
        self.btn_2d_resume.config(state="disabled")
        self.btn_2d_restart.config(state="normal")

        self.last_tick_time_2d = time.time()
        self._tick_2d_animation()

    def _open_2d_html(self):
        if os.path.exists(self.last_2d_html_path):
            webbrowser.open(f"file://{os.path.abspath(self.last_2d_html_path)}")
        else:
            fb = os.path.join(DIR_2D, "outputs", "intercept_2d_mode4_scenD.html")
            if os.path.exists(fb):
                webbrowser.open(f"file://{os.path.abspath(fb)}")
            else:
                messagebox.showinfo("Notice", "Please click 'Run Simulation' first to generate the web dashboard.")

    def _run_2d_monte_carlo(self):
        """Execute 2D Monte Carlo dispersion analysis from GUI."""
        try:
            self.status_var.set("Running 2D Monte Carlo dispersion analysis (20 runs)...")
            self.root.update_idletasks()

            mode = self.var_2d_mode.get()
            scen = self.var_2d_scen.get()
            target_kwargs = {}
            if scen == "LINE":
                target_kwargs = {"slope": 0.416667, "intercept": 0.0}
            elif scen == "PARABOLA":
                target_kwargs = {"a": 0.00005, "b": -0.2, "c": 1900.0}
            elif scen == "CUSTOM":
                target_kwargs = {
                    "x_expr": self.entry_2d_eq_x.get().strip() or "6000 - 250*t",
                    "y_expr": self.entry_2d_eq_y.get().strip() or "2500 + 400*sin(0.6*t)",
                }

            laws = [GuidanceLaw2D.TPN, GuidanceLaw2D.APN, GuidanceLaw2D.PP] if mode == 4 else (
                [GuidanceLaw2D.TPN] if mode == 1 else ([GuidanceLaw2D.APN] if mode == 2 else [GuidanceLaw2D.PP])
            )

            mc_sim = MonteCarloSimulator2D()
            mc_cfg = MonteCarloConfig2D(num_runs=20)

            report_sections = [
                "==========================================================================",
                f"       2D MONTE CARLO DISPERSION CAMPAIGN (20 RUNS) | Scenario: {scen}",
                "==========================================================================",
            ]
            for law in laws:
                summary = mc_sim.run_campaign(
                    guidance_law=law,
                    scenario_id=scen,
                    target_kwargs=target_kwargs,
                    mc_config=mc_cfg,
                )
                report_sections.append(summary.summary_table())

            self.txt_2d_summary.delete("1.0", tk.END)
            self.txt_2d_summary.insert(tk.END, "\n\n".join(report_sections))
            self.status_var.set("2D Monte Carlo campaign complete!")
        except Exception as e:
            messagebox.showerror("Monte Carlo Error", f"Error during Monte Carlo analysis:\n{str(e)}")
            self.status_var.set("Error during 2D Monte Carlo run.")

    # =========================================================================
    # SAFE WINDOW SHUTDOWN
    # =========================================================================
    def _on_window_close(self):
        """Safely cancel pending after() timers and release animation resources."""
        if hasattr(self, "anim_timer_3d") and self.anim_timer_3d:
            self.root.after_cancel(self.anim_timer_3d)
            self.anim_timer_3d = None
        if hasattr(self, "anim_timer_2d") and self.anim_timer_2d:
            self.root.after_cancel(self.anim_timer_2d)
            self.anim_timer_2d = None

        if hasattr(self, "anim_ctrl_3d") and self.anim_ctrl_3d:
            self.anim_ctrl_3d.stop()
        if hasattr(self, "anim_ctrl_2d") and self.anim_ctrl_2d:
            self.anim_ctrl_2d.stop()

        self.root.destroy()


def main():
    root = tk.Tk()
    app = AerospaceSimGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
