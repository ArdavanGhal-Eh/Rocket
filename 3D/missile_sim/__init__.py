"""
Missile-Target 3D Interception & Guidance Simulation Package
=============================================================
A high-fidelity aerospace simulation package modeling 3D Proportional Navigation (TPN)
and 3D Augmented Proportional Navigation (APN) against agile maneuvering fighter aircraft.
"""

from .config import MissileConfig, TargetConfig, SimulationConfig
from .guidance import GuidanceLaw, calculate_3d_tpn, calculate_3d_apn
from .target import (
    BaseTargetManeuver,
    StraightLineTarget,
    BarrelRollSpiralTarget,
    HighGInclinedTurnTarget,
    HighG3DSTurnTarget,
    create_target_maneuver,
)
from .missile import Missile3D
from .simulator import SimulationEngine, SimulationResult
from .fitting import TrajectoryFitter, PolynomialFitResult
from .visualizer import SimulationVisualizer

__all__ = [
    "MissileConfig",
    "TargetConfig",
    "SimulationConfig",
    "GuidanceLaw",
    "calculate_3d_tpn",
    "calculate_3d_apn",
    "BaseTargetManeuver",
    "StraightLineTarget",
    "BarrelRollSpiralTarget",
    "HighGInclinedTurnTarget",
    "HighG3DSTurnTarget",
    "create_target_maneuver",
    "Missile3D",
    "SimulationEngine",
    "SimulationResult",
    "TrajectoryFitter",
    "PolynomialFitResult",
    "SimulationVisualizer",
]
