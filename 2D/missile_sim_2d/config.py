"""
Configuration and Physical Parameters for 2D Interception Simulation.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class MissileConfig2D:
    """Physical, propulsion, and aerodynamic configuration of the missile in 2D."""

    # Mass properties (kg)
    m0: float = 85.0              # Initial total launch mass (kg)
    m_dry: float = 50.0           # Burnout dry mass (kg)
    m_propellant: float = 35.0    # Consumable propellant mass (kg)
    t_burn: float = 4.0           # Rocket motor burn time (s)
    mdot: float = 8.75            # Mass depletion rate (kg/s)

    # Propulsion properties (N)
    thrust_nominal: float = 17500.0  # Thrust force during burn (N)

    # Atmosphere & Aerodynamics
    rho: float = 0.736            # Atmospheric density at ~5000m (kg/m^3)
    diameter: float = 0.127       # Missile diameter (m)
    ref_area: float = 0.0127      # Cross-sectional area (m^2)
    cd_default: float = 0.40      # Aerodynamic drag coefficient

    # Kinematics and constraints
    v0_magnitude: float = 250.0   # Initial launch speed along LOS (m/s)
    r0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64))
    g_limit_lateral: float = 35.0 # Maximum lateral structural maneuverability in G (35G)
    g_accel: float = 9.81         # Gravity acceleration (m/s^2)
    include_gravity: bool = False # Set False for planar horizontal engagement at constant 5000m altitude

    # Guidance parameters
    nav_ratio: float = 4.0        # Navigation constant N = 4.0

    @property
    def max_lateral_accel(self) -> float:
        """Maximum lateral acceleration in m/s^2."""
        return self.g_limit_lateral * self.g_accel


@dataclass(frozen=True)
class TargetConfig2D:
    """Initial kinematic and flight properties of the fighter aircraft in 2D."""

    # Initial position (m) [Downrange X, Altitude/Crossrange Y]
    r0: np.ndarray = field(default_factory=lambda: np.array([6000.0, 2500.0], dtype=np.float64))

    # Cruise speed (m/s) ~ Mach 0.9 at 5000 m
    speed: float = 300.0

    # Initial heading unit vector (pointing towards origin)
    initial_heading: np.ndarray = field(
        default_factory=lambda: np.array([-6000.0, -2500.0], dtype=np.float64) / np.linalg.norm(np.array([6000.0, 2500.0]))
    )


@dataclass(frozen=True)
class SimulationConfig2D:
    """Numerical integration and simulation timing configuration."""

    dt: float = 0.005             # Integration time step (s)
    t_max: float = 25.0           # Maximum simulation horizon (s)
    speed_of_sound: float = 320.5 # Local speed of sound at 5000m altitude (m/s)
