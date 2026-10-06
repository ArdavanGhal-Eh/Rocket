"""
Missile-Target 2D Planar Interception & Guidance Simulation Package
===================================================================
A high-fidelity aerospace simulation package modeling 2D Proportional Navigation (TPN)
and 2D Augmented Proportional Navigation (APN) against agile maneuvering fighter aircraft.
"""

from .config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from .guidance import GuidanceLaw2D, calculate_2d_tpn, calculate_2d_apn
from .target import (
    BaseTargetManeuver2D,
    StraightLineTarget2D,
    SinusoidalWeaveTarget2D,
    HighGCircularTurnTarget2D,
    HighG2DSTurnTarget2D,
    create_target_maneuver_2d,
)
from .missile import Missile2D
from .simulator import SimulationEngine2D, SimulationResult2D
from .fitting import TrajectoryFitter2D, PolynomialFitResult2D
from .visualizer import SimulationVisualizer2D

__all__ = [
    "MissileConfig2D",
    "TargetConfig2D",
    "SimulationConfig2D",
    "GuidanceLaw2D",
    "calculate_2d_tpn",
    "calculate_2d_apn",
    "BaseTargetManeuver2D",
    "StraightLineTarget2D",
    "SinusoidalWeaveTarget2D",
    "HighGCircularTurnTarget2D",
    "HighG2DSTurnTarget2D",
    "create_target_maneuver_2d",
    "Missile2D",
    "SimulationEngine2D",
    "SimulationResult2D",
    "TrajectoryFitter2D",
    "PolynomialFitResult2D",
    "SimulationVisualizer2D",
]
