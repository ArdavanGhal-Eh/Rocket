"""
High-Fidelity 3D and 2D Analytical Visualization Suite.
======================================================
Produces:
1. Matplotlib 3D Trajectory plots with keyframe Line-of-Sight (LOS) vectors and impact blast points.
2. 2D Multi-panel analytical telemetry plots (Acceleration vs 35G limit, Mach number, Range, LOS rate, Control Energy).
3. Interactive Plotly 3D HTML dashboard with full 360-degree rotation, pan, zoom, and rich telemetry hover tooltips.
"""

from typing import List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from .simulator import SimulationResult
from .guidance import GuidanceLaw


class SimulationVisualizer:
    """
    Visualization manager for missile-target engagement results.
    """

    def __init__(self, style_dark: bool = False):
        self.style_dark = style_dark
        if style_dark:
            plt.style.use("dark_background")
        else:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_trajectories_3d_matplotlib(
        self,
        results: List[SimulationResult],
        title: str = "3D Missile Interception Trajectory",
        num_los_rays: int = 6,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Render publication-grade 3D trajectory plot in Matplotlib.
        """
        fig = plt.figure(figsize=(12, 9), dpi=120)
        ax = fig.add_subplot(111, projection="3d")

        # Color palettes for different guidance laws
        colors = {
            GuidanceLaw.TPN: "#1f77b4",  # Steel Blue
            GuidanceLaw.APN: "#2ca02c",  # Emerald Green
            GuidanceLaw.PP: "#d97706",   # Amber / Orange
        }
        styles = {
            GuidanceLaw.TPN: "-",
            GuidanceLaw.APN: "-.",
            GuidanceLaw.PP: ":",
        }

        # Plot target trajectory for longest simulated flight duration
        max_idx = int(np.argmax([len(r.time) for r in results]))
        res0 = results[0]
        r_T = results[max_idx].r_T
        ax.plot(
            r_T[:, 0], r_T[:, 1], r_T[:, 2],
            color="#d62728", lw=2.5, label=f"Target: {res0.target_name}", linestyle="--"
        )
        # Target start
        ax.scatter(
            [r_T[0, 0]], [r_T[0, 1]], [r_T[0, 2]],
            color="#d62728", marker="^", s=100, label="Target Start"
        )

        # Plot each missile trajectory
        for res in results:
            c = colors.get(res.law, "#9467bd")
            st = styles.get(res.law, "-")
            if res.law == GuidanceLaw.TPN:
                law_lbl = "3D TPN"
            elif res.law == GuidanceLaw.APN:
                law_lbl = "3D APN"
            else:
                law_lbl = "3D Pure Pursuit"
            label_name = f"Missile ({law_lbl})"
            ax.plot(
                res.r_M[:, 0], res.r_M[:, 1], res.r_M[:, 2],
                color=c, lw=2.5, linestyle=st, label=label_name
            )

            # Draw keyframe LOS rays for primary trajectory
            if num_los_rays > 0:
                indices = np.linspace(0, len(res.time) - 1, num_los_rays, dtype=int)
                for idx in indices:
                    rm = res.r_M[idx]
                    rt = res.r_T[idx]
                    ax.plot(
                        [rm[0], rt[0]], [rm[1], rt[1]], [rm[2], rt[2]],
                        color=c, lw=0.9, linestyle=":", alpha=0.45
                    )

            # Impact point
            ax.scatter(
                [res.hit_location_missile[0]],
                [res.hit_location_missile[1]],
                [res.hit_location_missile[2]],
                color=c, marker="*", s=260,
                edgecolors="black",
                label=f"Impact ({res.law.name}): Miss={res.miss_distance:.3f}m, t={res.intercept_time:.2f}s"
            )

        # Launch point
        ax.scatter([0], [0], [0], color="#17becf", marker="o", s=110, label="Launch Origin [0, 0, 0]")

        # Axis labeling and formatting
        ax.set_xlabel("Downrange X (m)", fontsize=11, labelpad=10)
        ax.set_ylabel("Crossrange Y (m)", fontsize=11, labelpad=10)
        ax.set_zlabel("Altitude Z (m)", fontsize=11, labelpad=10)
        ax.set_title(f"{title}\nTarget Maneuver: {res0.target_name}", fontsize=13, fontweight="bold", pad=15)
        ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1), fontsize=9, frameon=True)
        ax.view_init(elev=26, azim=-55)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Matplotlib 3D] Saved 3D trajectory plot to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_telemetry_2d(
        self,
        results: List[SimulationResult],
        title: str = "Engagement Telemetry Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Render 4-panel analytical time-series graphs:
        1. Commanded / Lateral acceleration vs 35G limit
        2. Missile velocity and Mach number (showing boost vs coast phases)
        3. Range to target & Line-of-sight angular rate
        4. Cumulative guidance control energy integral
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
        colors = {
            GuidanceLaw.TPN: "#1f77b4",
            GuidanceLaw.APN: "#2ca02c",
            GuidanceLaw.PP: "#d97706",
        }

        def _get_law_name(l):
            if l == GuidanceLaw.TPN:
                return "3D TPN"
            elif l == GuidanceLaw.APN:
                return "3D APN"
            return "3D Pure Pursuit"

        # -------------------------------------------------------------
        # Panel 1: Lateral Acceleration vs 35G Limit & Total Net Accel
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        for res in results:
            c = colors.get(res.law, "blue")
            lname = _get_law_name(res.law)
            label_lat = f"{lname} Steering G (Peak={res.peak_lateral_g:.1f}G)"
            ax1.plot(res.time, res.a_lateral_g, color=c, lw=2.0, label=label_lat)
            label_net = f"{lname} Total G"
            ax1.plot(res.time, res.a_net_g, color=c, lw=1.2, linestyle=":", alpha=0.55, label=label_net)

        # 35G limit line
        ax1.axhline(35.0, color="crimson", linestyle="--", lw=1.8, label="Max Lateral G Limit (35G)")
        ax1.set_title("Missile Steering & Total Accel vs 35G Structural Limit", fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Acceleration (G)")
        ax1.set_ylim(0, 50)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 2: Speed and Mach Number (Boost / Coast Phases)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        for res in results:
            c = colors.get(res.law, "blue")
            lname = _get_law_name(res.law)
            lbl_speed = f"{lname} Speed"
            ax2.plot(res.time, res.speed_M, color=c, lw=2.0, label=lbl_speed)

        # Mark motor burnout at t = 4.0s
        ax2.axvline(4.0, color="orange", linestyle=":", lw=2.0, label="Burnout (t = 4.0s)")
        ax2.set_title("Missile Velocity & Mach Profile (Boost vs Coast)", fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Missile Speed (m/s)")

        # Secondary axis for Mach
        ax2_mach = ax2.twinx()
        for res in results:
            c = colors.get(res.law, "blue")
            ax2_mach.plot(res.time, res.mach_M, color=c, lw=1.0, linestyle="--", alpha=0.5)
        ax2_mach.set_ylabel("Mach Number (a = 320.5 m/s)", color="gray")
        ax2_mach.grid(False)
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 3: Relative Range & LOS Angular Rate
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        for res in results:
            c = colors.get(res.law, "blue")
            lname = _get_law_name(res.law)
            lbl_rng = f"{lname} Range"
            ax3.plot(res.time, res.range_dist, color=c, lw=2.0, label=lbl_rng)

        ax3.set_title("Range to Target and LOS Angular Velocity Rate", fontweight="bold")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Relative Range (m)")

        # Secondary axis for LOS rate |Omega_LOS|
        ax3_los = ax3.twinx()
        for res in results:
            c = colors.get(res.law, "blue")
            lname = _get_law_name(res.law)
            lbl_omega = f"{lname} |" + r"$\Omega_{LOS}$|"
            ax3_los.plot(res.time, np.degrees(res.omega_los_mag), color=c, lw=1.5, linestyle=":", alpha=0.8, label=lbl_omega)
        ax3_los.set_ylabel("LOS Rate (deg/s)", color="purple")
        ax3_los.grid(False)

        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 4: Cumulative Guidance Control Energy
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        for res in results:
            c = colors.get(res.law, "blue")
            lname = _get_law_name(res.law)
            lbl_energy = f"{lname} (Total={res.total_control_energy:.1e})"
            ax4.plot(res.time, res.control_energy, color=c, lw=2.0, label=lbl_energy)

        ax4.set_title(r"Control Energy Consumption Integral $\int |a_{cmd}|^2 dt$", fontweight="bold")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Control Effort ($m^2 / s^3$)")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle(f"{title} - [{results[0].target_name}]", fontsize=14, fontweight="bold", y=0.99)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Matplotlib 2D] Saved 2D telemetry plot to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def generate_interactive_plotly_3d(
        self,
        results: List[SimulationResult],
        title: str = "3D Interactive Interception Engagement Dashboard",
        output_html: str = "intercept_simulation_3d.html",
        num_los_rays: int = 8,
    ) -> str:
        """
        Generate a fully interactive 3D WebGL / HTML visualization using Plotly.
        Allows complete 360-degree rotation, zooming, pan, and detailed telemetry hovers.
        """
        fig = go.Figure()

        colors = {
            GuidanceLaw.TPN: "#1f77b4",
            GuidanceLaw.APN: "#2ca02c",
            GuidanceLaw.PP: "#d97706",
        }

        # 1. Target Trajectory
        res0 = results[0]
        r_T = res0.r_T
        target_hover = [
            f"<b>Target (Fighter)</b><br>Time: {t:.2f}s<br>Pos: [{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}] m<br>Speed: {np.linalg.norm(v):.1f} m/s<br>Accel: {np.linalg.norm(a)/9.81:.1f} G"
            for t, p, v, a in zip(res0.time, res0.r_T, res0.v_T, res0.a_T)
        ]

        fig.add_trace(
            go.Scatter3d(
                x=r_T[:, 0], y=r_T[:, 1], z=r_T[:, 2],
                mode="lines",
                line=dict(color="#d62728", width=5, dash="dash"),
                name=f"Target: {res0.target_name}",
                hoverinfo="text",
                text=target_hover,
            )
        )

        # Target start point
        fig.add_trace(
            go.Scatter3d(
                x=[r_T[0, 0]], y=[r_T[0, 1]], z=[r_T[0, 2]],
                mode="markers+text",
                marker=dict(size=8, color="#d62728", symbol="diamond"),
                name="Target Initial Position",
                text=["Target Start"],
                textposition="top center",
            )
        )

        # 2. Missile Trajectories
        for res in results:
            c = colors.get(res.law, "#9467bd")
            if res.law == GuidanceLaw.TPN:
                law_label = "3D TPN"
            elif res.law == GuidanceLaw.APN:
                law_label = "3D APN"
            else:
                law_label = "3D Pure Pursuit"

            missile_hover = [
                f"<b>Missile ({law_label})</b><br>Time: {t:.2f}s<br>Pos: [{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}] m<br>Speed: {spd:.1f} m/s (Mach {mch:.2f})<br>Lat G: {g:.1f} G<br>Range: {rng:.1f} m"
                for t, p, spd, mch, g, rng in zip(
                    res.time, res.r_M, res.speed_M, res.mach_M, res.a_lateral_g, res.range_dist
                )
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=res.r_M[:, 0], y=res.r_M[:, 1], z=res.r_M[:, 2],
                    mode="lines",
                    line=dict(color=c, width=6),
                    name=f"Missile Trajectory ({law_label})",
                    hoverinfo="text",
                    text=missile_hover,
                )
            )

            # Keyframe Line-of-sight rays
            if num_los_rays > 0:
                indices = np.linspace(0, len(res.time) - 1, num_los_rays, dtype=int)
                for idx in indices:
                    rm = res.r_M[idx]
                    rt = res.r_T[idx]
                    fig.add_trace(
                        go.Scatter3d(
                            x=[rm[0], rt[0]], y=[rm[1], rt[1]], z=[rm[2], rt[2]],
                            mode="lines",
                            line=dict(color=c, width=1.5, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            # Impact point
            fig.add_trace(
                go.Scatter3d(
                    x=[res.hit_location_missile[0]],
                    y=[res.hit_location_missile[1]],
                    z=[res.hit_location_missile[2]],
                    mode="markers+text",
                    marker=dict(size=12, color=c, symbol="cross"),
                    name=f"Intercept ({law_label}): Miss={res.miss_distance:.2f}m",
                    text=[f"Impact ({law_label})<br>t={res.intercept_time:.2f}s"],
                    textposition="bottom center",
                )
            )

        # Launch Origin
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                marker=dict(size=9, color="#17becf", symbol="circle"),
                name="Launch Point [0, 0, 0]",
                text=["Launch [0,0,0]"],
                textposition="top right",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sup>Target Maneuver: {res0.target_name} | Atmosphere: rho=0.736 kg/m^3 (5000m)</sup>",
                x=0.5,
                font=dict(size=16),
            ),
            scene=dict(
                xaxis=dict(title="Downrange X (m)", backgroundcolor="#f0f2f6", gridcolor="white"),
                yaxis=dict(title="Crossrange Y (m)", backgroundcolor="#f0f2f6", gridcolor="white"),
                zaxis=dict(title="Altitude Z (m)", backgroundcolor="#f0f2f6", gridcolor="white"),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=-1.5, y=-1.8, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            margin=dict(l=10, r=10, b=10, t=60),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        )

        abs_html_path = os.path.abspath(output_html)
        os.makedirs(os.path.dirname(abs_html_path), exist_ok=True)
        fig.write_html(abs_html_path)
        print(f"[Plotly 3D] Saved interactive WebGL dashboard to: {abs_html_path}")
        return abs_html_path
