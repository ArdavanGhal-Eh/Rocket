"""
====================================================================================================
Aerospace Simulation Suite | Animation Controller & Real-Time Visualization Engine
====================================================================================================
Provides decoupled, robust playback orchestration and frame-by-frame rendering for 2D and 3D
missile-target engagements without blocking or freezing the GUI event loop.
Supports Pause, Resume, Restart, Variable Playback Speeds, and Live Telemetry State Tracking.
====================================================================================================
"""

import enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


class AnimationState(enum.Enum):
    """Lifecycle states of the simulation animation controller."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class MissileFrameState:
    """Instantaneous telemetry and kinematic state of a missile for a single animation frame."""
    name: str
    law: Any
    r_M: np.ndarray           # Position coordinates [x, y] or [x, y, z]
    v_M: np.ndarray           # Velocity vector
    speed: float              # Scalar speed (m/s)
    mach: float               # Mach number
    range_dist: float         # Distance to target (m)
    closing_speed: float      # Closing velocity Vc (m/s)
    lateral_g: float          # Commanded/lateral steering acceleration (G)
    is_saturated: bool        # Structural 35G saturation flag
    control_energy: float     # Cumulative control energy integral (m^2/s^3)


@dataclass
class PlaybackStateData:
    """Comprehensive snapshot of the engagement for the current animation frame."""
    sim_time: float
    frame_index: int
    total_frames: int
    progress_fraction: float
    state: AnimationState
    is_terminal: bool
    r_T: np.ndarray                           # Target current position
    v_T: np.ndarray                           # Target current velocity
    missile_states: List[MissileFrameState]   # States of all active missiles


class AnimationPlaybackController:
    """
    Decoupled playback controller managing simulation playback timing, frame scheduling,
    interpolation, speed throttling, and state transitions. Fully testable without a GUI.
    """

    SUPPORTED_SPEEDS = [0.25, 0.5, 1.0, 2.0, 5.0]

    def __init__(self, speed_multiplier: float = 1.0):
        self.speed_multiplier = float(speed_multiplier)
        self.state = AnimationState.IDLE
        self.results: List[Any] = []
        self.time_array: np.ndarray = np.array([], dtype=np.float64)
        self.current_sim_time: float = 0.0
        self.current_frame_idx: int = 0
        self.dt_nominal: float = 0.005
        self.error_message: Optional[str] = None

    def is_running(self) -> bool:
        return self.state == AnimationState.RUNNING

    def is_paused(self) -> bool:
        return self.state == AnimationState.PAUSED

    def is_completed(self) -> bool:
        return self.state == AnimationState.COMPLETED

    def load_results(self, results: List[Any]) -> None:
        """
        Validate and load simulation results from 2D or 3D simulation engines.
        Raises ValueError if results are empty, invalid, or contain unusable trajectory data.
        """
        if not results:
            self.state = AnimationState.ERROR
            self.error_message = "No simulation results provided."
            raise ValueError(self.error_message)

        # Validate first result
        res0 = results[0]
        if not hasattr(res0, "time") or not hasattr(res0, "r_M") or not hasattr(res0, "r_T"):
            self.state = AnimationState.ERROR
            self.error_message = "Simulation result missing required time or trajectory fields."
            raise ValueError(self.error_message)

        if len(res0.time) == 0 or len(res0.r_M) == 0 or len(res0.r_T) == 0:
            self.state = AnimationState.ERROR
            self.error_message = "Simulation trajectory contains zero data frames."
            raise ValueError(self.error_message)

        # Check for NaN / Inf in critical trajectory arrays
        for r in results:
            if np.any(np.isnan(r.r_M)) or np.any(np.isinf(r.r_M)):
                self.state = AnimationState.ERROR
                self.error_message = "Missile trajectory contains NaN or Inf values."
                raise ValueError(self.error_message)
            if np.any(np.isnan(r.r_T)) or np.any(np.isinf(r.r_T)):
                self.state = AnimationState.ERROR
                self.error_message = "Target trajectory contains NaN or Inf values."
                raise ValueError(self.error_message)

        # Reference time vector: choose result with maximum duration
        max_idx = int(np.argmax([len(r.time) for r in results]))
        self.results = results
        self.time_array = results[max_idx].time

        if len(self.time_array) > 1:
            self.dt_nominal = float(self.time_array[1] - self.time_array[0])
            if self.dt_nominal <= 0:
                self.dt_nominal = 0.005
        else:
            self.dt_nominal = 0.005

        self.current_sim_time = float(self.time_array[0])
        self.current_frame_idx = 0
        self.state = AnimationState.IDLE
        self.error_message = None

    def start(self) -> None:
        """Initiate playback from the first frame."""
        if not self.results or len(self.time_array) == 0:
            raise RuntimeError("Cannot start animation: No results loaded.")
        self.current_sim_time = float(self.time_array[0])
        self.current_frame_idx = 0
        self.state = AnimationState.RUNNING

    def pause(self) -> None:
        """Pause playback at the current frame."""
        if self.state == AnimationState.RUNNING:
            self.state = AnimationState.PAUSED

    def resume(self) -> None:
        """Resume playback from the current paused frame."""
        if self.state == AnimationState.PAUSED:
            self.state = AnimationState.RUNNING

    def restart(self) -> None:
        """Restart playback from frame 0."""
        if not self.results or len(self.time_array) == 0:
            raise RuntimeError("Cannot restart animation: No results loaded.")
        self.current_sim_time = float(self.time_array[0])
        self.current_frame_idx = 0
        self.state = AnimationState.RUNNING

    def stop(self) -> None:
        """Stop animation and return to IDLE state."""
        self.state = AnimationState.IDLE

    def set_speed(self, speed: float) -> None:
        """Adjust animation rendering playback speed multiplier (e.g. 0.25x to 5x)."""
        if speed <= 0:
            raise ValueError(f"Speed multiplier must be positive, got {speed}")
        self.speed_multiplier = float(speed)

    def step(self, real_elapsed_seconds: float) -> PlaybackStateData:
        """
        Advance animation time according to real elapsed time and playback speed multiplier.
        Returns the new PlaybackStateData snapshot.
        """
        if not self.results or len(self.time_array) == 0:
            raise RuntimeError("Cannot step animation: No results loaded.")

        if self.state == AnimationState.RUNNING:
            sim_dt_step = max(0.0, real_elapsed_seconds) * self.speed_multiplier
            self.current_sim_time += sim_dt_step

            # Find matching frame index via searchsorted
            idx = int(np.searchsorted(self.time_array, self.current_sim_time))
            if idx >= len(self.time_array) - 1:
                idx = len(self.time_array) - 1
                self.current_sim_time = float(self.time_array[-1])
                self.state = AnimationState.COMPLETED

            self.current_frame_idx = idx

        return self.get_current_data()

    def get_current_data(self) -> PlaybackStateData:
        """Construct PlaybackStateData for the current frame index."""
        if not self.results or len(self.time_array) == 0:
            raise RuntimeError("No simulation data available.")

        idx = self.current_frame_idx
        total_frames = len(self.time_array)
        t_cur = float(self.time_array[idx]) if idx < total_frames else float(self.time_array[-1])
        progress = float(idx / (total_frames - 1)) if total_frames > 1 else 1.0
        is_term = (idx >= total_frames - 1) or (self.state == AnimationState.COMPLETED)

        # Target position from longest run
        max_idx = int(np.argmax([len(r.time) for r in self.results]))
        res_target = self.results[max_idx]
        t_idx = min(idx, len(res_target.r_T) - 1)
        r_T = res_target.r_T[t_idx]
        v_T = res_target.v_T[t_idx]

        missile_states: List[MissileFrameState] = []
        for r in self.results:
            # Clamp index to specific missile flight duration (in case one intercepted earlier)
            m_idx = min(idx, len(r.time) - 1)
            law_name = getattr(r.law, "name", str(r.law))
            m_state = MissileFrameState(
                name=f"Missile ({law_name})",
                law=r.law,
                r_M=r.r_M[m_idx],
                v_M=r.v_M[m_idx],
                speed=float(r.speed_M[m_idx]),
                mach=float(r.mach_M[m_idx]),
                range_dist=float(r.range_dist[m_idx]),
                closing_speed=float(r.closing_speed[m_idx]),
                lateral_g=float(r.a_lateral_g[m_idx]),
                is_saturated=bool(r.is_saturated[m_idx]),
                control_energy=float(r.control_energy[m_idx]),
            )
            missile_states.append(m_state)

        return PlaybackStateData(
            sim_time=t_cur,
            frame_index=idx,
            total_frames=total_frames,
            progress_fraction=progress,
            state=self.state,
            is_terminal=is_term,
            r_T=r_T,
            v_T=v_T,
            missile_states=missile_states,
        )


class MatplotlibAnimationRenderer2D:
    """
    High-performance, flicker-free Matplotlib 2D Planar renderer for live GUI animation.
    Initializes axes with proper auto-scaling and padding, then updates line/marker data
    in place without expensive re-plots or canvas clearing.
    """

    COLOR_MAP = {
        "TPN": "#1f77b4",       # Blue
        "APN": "#2ca02c",       # Green
        "PP": "#d97706",        # Orange / Amber
        "PURE_PURSUIT": "#d97706",
    }

    def __init__(self):
        self.ax = None
        self.target_trail = None
        self.target_marker = None
        self.missile_trails: List[Any] = []
        self.missile_markers: List[Any] = []
        self.los_lines: List[Any] = []
        self.impact_markers: List[Any] = []
        self.results: List[Any] = []

    def setup(self, ax, results: List[Any], target_name: str = "Target"):
        """Prepare axes, calculate bounds with 8% padding, and create dynamic visual artists."""
        self.ax = ax
        self.results = results
        self.ax.clear()

        # Compute combined bounding box for all missiles and target across full engagement
        max_idx = int(np.argmax([len(r.time) for r in results]))
        r_T_full = results[max_idx].r_T

        all_x = [r_T_full[:, 0]]
        all_y = [r_T_full[:, 1]]
        for r in results:
            all_x.append(r.r_M[:, 0])
            all_y.append(r.r_M[:, 1])

        concat_x = np.concatenate(all_x)
        concat_y = np.concatenate(all_y)

        x_min, x_max = float(np.min(concat_x)), float(np.max(concat_x))
        y_min, y_max = float(np.min(concat_y)), float(np.max(concat_y))

        # Add 8-10% padding so neither missile nor target ever touches or leaves viewport
        span_x = max(x_max - x_min, 500.0)
        span_y = max(y_max - y_min, 500.0)
        pad_x = span_x * 0.08
        pad_y = span_y * 0.08

        self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax.set_ylim(y_min - pad_y, y_max + pad_y)

        # Faint ghost/guidance path for full target trajectory (alpha=0.18)
        self.ax.plot(
            r_T_full[:, 0], r_T_full[:, 1],
            color="#e53e3e", lw=1.2, linestyle="--", alpha=0.22,
            label=f"Target Route: {target_name}"
        )

        # Launch origin [0, 0]
        self.ax.scatter([0], [0], color="#00b4d8", marker="o", s=85, edgecolors="#0077b6", lw=1.5, label="Launch Origin [0, 0]", zorder=4)

        # Dynamic target trail and marker
        self.target_trail, = self.ax.plot([], [], color="#e53e3e", lw=2.4, linestyle="--", label="Target Trail", zorder=3)
        self.target_marker, = self.ax.plot([], [], color="#e53e3e", marker="^", markersize=11, markeredgecolor="#742a2a", markeredgewidth=1.2, linestyle="None", label="Target (Live)", zorder=6)

        # Dynamic missile trails and markers
        self.missile_trails = []
        self.missile_markers = []
        self.los_lines = []
        self.impact_markers = []

        for r in results:
            law_key = getattr(r.law, "name", str(r.law)).upper()
            color = "#1f77b4"
            for k, c in self.COLOR_MAP.items():
                if k in law_key:
                    color = c
                    break

            trail, = self.ax.plot([], [], color=color, lw=2.2, label=f"Missile ({r.law.name if hasattr(r.law, 'name') else r.law})", zorder=4)
            marker, = self.ax.plot([], [], color=color, marker="o", markersize=9, markeredgecolor="black", markeredgewidth=1.0, linestyle="None", zorder=7)
            los, = self.ax.plot([], [], color=color, linestyle=":", lw=1.0, alpha=0.55, zorder=2)

            self.missile_trails.append(trail)
            self.missile_markers.append(marker)
            self.los_lines.append(los)

        self.ax.set_xlabel("Downrange X (m)", fontsize=9, fontweight="bold")
        self.ax.set_ylabel("Crossrange / Altitude Y (m)", fontsize=9, fontweight="bold")
        self.ax.set_title(f"Planar 2D Live Interception Animation | Target: {target_name}", fontsize=10, fontweight="bold")
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    def update_frame(self, state: PlaybackStateData):
        """Update line trails, object markers, and LOS rays in place."""
        if not self.ax or not self.results:
            return

        idx = state.frame_index
        max_idx = int(np.argmax([len(r.time) for r in self.results]))
        r_T_full = self.results[max_idx].r_T

        # Update target trail and marker
        t_idx = min(idx + 1, len(r_T_full))
        self.target_trail.set_data(r_T_full[:t_idx, 0], r_T_full[:t_idx, 1])
        self.target_marker.set_data([state.r_T[0]], [state.r_T[1]])

        # Update each missile trail, marker, and line-of-sight ray
        for i, r in enumerate(self.results):
            m_idx = min(idx + 1, len(r.r_M))
            cur_m_idx = min(idx, len(r.r_M) - 1)
            r_M_cur = r.r_M[cur_m_idx]

            self.missile_trails[i].set_data(r.r_M[:m_idx, 0], r.r_M[:m_idx, 1])
            self.missile_markers[i].set_data([r_M_cur[0]], [r_M_cur[1]])

            # Line of Sight between missile and target
            self.los_lines[i].set_data([r_M_cur[0], state.r_T[0]], [r_M_cur[1], state.r_T[1]])

    def finalize(self):
        """Draw final terminal blast/impact stars and closest point of approach indicators."""
        if not self.ax or not self.results:
            return

        for r in self.results:
            law_key = getattr(r.law, "name", str(r.law)).upper()
            color = "#1f77b4"
            for k, c in self.COLOR_MAP.items():
                if k in law_key:
                    color = c
                    break

            self.ax.scatter(
                [r.hit_location_missile[0]], [r.hit_location_missile[1]],
                color=color, marker="*", s=220, edgecolors="black", lw=1.2, zorder=10,
                label=f"Terminal Impact: Miss={r.miss_distance:.3f}m, t={r.intercept_time:.2f}s"
            )

        self.ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)


class MatplotlibAnimationRenderer3D:
    """
    High-performance Matplotlib 3D Spatial renderer for live GUI animation.
    Updates 3D line data and scatter markers while preserving interactive orbit/mouse rotations.
    """

    COLOR_MAP = {
        "TPN": "#1f77b4",       # Blue
        "APN": "#2ca02c",       # Green
        "PP": "#d97706",        # Orange
        "PURE_PURSUIT": "#d97706",
    }

    def __init__(self):
        self.ax = None
        self.target_trail = None
        self.target_marker = None
        self.missile_trails: List[Any] = []
        self.missile_markers: List[Any] = []
        self.los_lines: List[Any] = []
        self.results: List[Any] = []

    def setup(self, ax, results: List[Any], target_name: str = "Target"):
        """Prepare 3D spatial viewport, calculate spatial bounds, and create artists."""
        self.ax = ax
        self.results = results
        self.ax.clear()

        max_idx = int(np.argmax([len(r.time) for r in results]))
        r_T_full = results[max_idx].r_T

        all_x = [r_T_full[:, 0]]
        all_y = [r_T_full[:, 1]]
        all_z = [r_T_full[:, 2]]
        for r in results:
            all_x.append(r.r_M[:, 0])
            all_y.append(r.r_M[:, 1])
            all_z.append(r.r_M[:, 2])

        concat_x = np.concatenate(all_x)
        concat_y = np.concatenate(all_y)
        concat_z = np.concatenate(all_z)

        x_min, x_max = float(np.min(concat_x)), float(np.max(concat_x))
        y_min, y_max = float(np.min(concat_y)), float(np.max(concat_y))
        z_min, z_max = float(np.min(concat_z)), float(np.max(concat_z))

        span_x = max(x_max - x_min, 500.0)
        span_y = max(y_max - y_min, 500.0)
        span_z = max(z_max - z_min, 500.0)

        pad_x = span_x * 0.08
        pad_y = span_y * 0.08
        pad_z = span_z * 0.08

        self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax.set_ylim(y_min - pad_y, y_max + pad_y)
        self.ax.set_zlim(z_min - pad_z, z_max + pad_z)

        # Ghost route for target
        self.ax.plot(
            r_T_full[:, 0], r_T_full[:, 1], r_T_full[:, 2],
            color="#e53e3e", lw=1.2, linestyle="--", alpha=0.25,
            label=f"Target Route: {target_name}"
        )

        # Launch Origin [0, 0, 0]
        self.ax.scatter([0], [0], [0], color="#00b4d8", marker="o", s=80, edgecolors="#0077b6", label="Launch Origin [0, 0, 0]")

        # Dynamic target trail and marker
        self.target_trail, = self.ax.plot([], [], [], color="#e53e3e", lw=2.4, linestyle="--", label="Target Trail")
        self.target_marker, = self.ax.plot([], [], [], color="#e53e3e", marker="^", markersize=10, markeredgecolor="#742a2a", linestyle="None", label="Target (Live)")

        self.missile_trails = []
        self.missile_markers = []
        self.los_lines = []

        for r in results:
            law_key = getattr(r.law, "name", str(r.law)).upper()
            color = "#1f77b4"
            for k, c in self.COLOR_MAP.items():
                if k in law_key:
                    color = c
                    break

            trail, = self.ax.plot([], [], [], color=color, lw=2.2, label=f"Missile ({r.law.name if hasattr(r.law, 'name') else r.law})")
            marker, = self.ax.plot([], [], [], color=color, marker="o", markersize=9, markeredgecolor="black", linestyle="None")
            los, = self.ax.plot([], [], [], color=color, linestyle=":", lw=1.0, alpha=0.5)

            self.missile_trails.append(trail)
            self.missile_markers.append(marker)
            self.los_lines.append(los)

        self.ax.set_xlabel("Downrange X (m)", fontsize=8, fontweight="bold")
        self.ax.set_ylabel("Crossrange Y (m)", fontsize=8, fontweight="bold")
        self.ax.set_zlabel("Altitude Z (m)", fontsize=8, fontweight="bold")
        self.ax.set_title(f"3D Spatial Live Interception Animation | Target: {target_name}", fontsize=10, fontweight="bold")
        self.ax.view_init(elev=26, azim=-55)
        self.ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    def update_frame(self, state: PlaybackStateData):
        """Update 3D trajectory trails, markers, and line-of-sight vectors."""
        if not self.ax or not self.results:
            return

        idx = state.frame_index
        max_idx = int(np.argmax([len(r.time) for r in self.results]))
        r_T_full = self.results[max_idx].r_T

        t_idx = min(idx + 1, len(r_T_full))
        self.target_trail.set_data(r_T_full[:t_idx, 0], r_T_full[:t_idx, 1])
        self.target_trail.set_3d_properties(r_T_full[:t_idx, 2])

        self.target_marker.set_data([state.r_T[0]], [state.r_T[1]])
        self.target_marker.set_3d_properties([state.r_T[2]])

        for i, r in enumerate(self.results):
            m_idx = min(idx + 1, len(r.r_M))
            cur_m_idx = min(idx, len(r.r_M) - 1)
            r_M_cur = r.r_M[cur_m_idx]

            self.missile_trails[i].set_data(r.r_M[:m_idx, 0], r.r_M[:m_idx, 1])
            self.missile_trails[i].set_3d_properties(r.r_M[:m_idx, 2])

            self.missile_markers[i].set_data([r_M_cur[0]], [r_M_cur[1]])
            self.missile_markers[i].set_3d_properties([r_M_cur[2]])

            self.los_lines[i].set_data([r_M_cur[0], state.r_T[0]], [r_M_cur[1], state.r_T[1]])
            self.los_lines[i].set_3d_properties([r_M_cur[2], state.r_T[2]])

    def finalize(self):
        """Place final 3D impact markers."""
        if not self.ax or not self.results:
            return

        for r in self.results:
            law_key = getattr(r.law, "name", str(r.law)).upper()
            color = "#1f77b4"
            for k, c in self.COLOR_MAP.items():
                if k in law_key:
                    color = c
                    break

            self.ax.scatter(
                [r.hit_location_missile[0]], [r.hit_location_missile[1]], [r.hit_location_missile[2]],
                color=color, marker="*", s=220, edgecolors="black", lw=1.2,
                label=f"Impact: Miss={r.miss_distance:.3f}m, t={r.intercept_time:.2f}s"
            )

        self.ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
