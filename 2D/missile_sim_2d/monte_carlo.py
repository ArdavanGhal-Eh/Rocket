"""
Monte Carlo Dispersion Analysis Engine for 2D Planar Missile Interception.
==========================================================================
Executes stochastic sensitivity and dispersion campaigns in 2D to evaluate:
- Circular Error Probable (CEP - 50% percentile miss distance)
- R95 (95% containment radius)
- Probability of Kill / Intercept (Pk)
- Parametric variances in missile launch conditions and maneuvering target parameters
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any
import numpy as np

from .config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from .guidance import GuidanceLaw2D
from .target import create_target_maneuver_2d, BaseTargetManeuver2D
from .simulator import SimulationEngine2D, SimulationResult2D


@dataclass
class MonteCarloConfig2D:
    """Configuration parameters for 2D stochastic Monte Carlo dispersion campaigns."""
    num_runs: int = 50
    random_seed: Optional[int] = 42
    lethal_radius: float = 2.0
    sigma_v0_speed: float = 4.0
    sigma_target_speed: float = 6.0
    sigma_target_r0: float = 40.0
    sigma_thrust_pct: float = 0.02


@dataclass
class MonteCarloRunResult2D:
    """Telemetry summary for an individual 2D Monte Carlo simulation run."""
    run_id: int
    miss_distance: float
    intercept_time: float
    peak_lateral_g: float
    total_control_energy: float
    is_kill: bool


@dataclass
class MonteCarloSummary2D:
    """Aggregated statistical metrics across all 2D Monte Carlo campaign iterations."""
    law: GuidanceLaw2D
    target_scenario: str
    total_runs: int
    kill_probability_pk: float
    mean_miss_distance: float
    std_miss_distance: float
    median_miss_distance: float
    cep_50: float
    r95: float
    mean_flight_time: float
    mean_control_energy: float
    mean_peak_g: float
    runs: List[MonteCarloRunResult2D] = field(default_factory=list)

    def summary_table(self) -> str:
        """Render a formatted ASCII statistical summary of the 2D dispersion campaign."""
        lines = [
            "=" * 85,
            f"  2D MONTE CARLO DISPERSION ANALYSIS ({self.total_runs} RUNS) | {self.law.value}",
            f"  Target Scenario: {self.target_scenario}",
            "=" * 85,
            f"Probability of Kill (Pk < {2.0}m) : {self.kill_probability_pk * 100:.1f} %",
            f"Circular Error Probable (CEP 50%)  : {self.cep_50:.4f} m",
            f"95% Containment Radius (R95)       : {self.r95:.4f} m",
            f"Mean Miss Distance ± Std Dev       : {self.mean_miss_distance:.4f} ± {self.std_miss_distance:.4f} m",
            f"Median Miss Distance               : {self.median_miss_distance:.4f} m",
            "-" * 85,
            f"Average Time of Flight             : {self.mean_flight_time:.2f} s",
            f"Average Peak Lateral Load          : {self.mean_peak_g:.2f} G",
            f"Average Control Energy (m²/s³)     : {self.mean_control_energy:.2e}",
            "=" * 85,
        ]
        return "\n".join(lines)


class MonteCarloSimulator2D:
    """
    Executes stochastic 2D dispersion campaigns for missile guidance algorithms.
    """

    def __init__(
        self,
        base_missile_cfg: Optional[MissileConfig2D] = None,
        base_target_cfg: Optional[TargetConfig2D] = None,
        base_sim_cfg: Optional[SimulationConfig2D] = None,
    ):
        self.base_missile_cfg = base_missile_cfg or MissileConfig2D()
        self.base_target_cfg = base_target_cfg or TargetConfig2D()
        self.base_sim_cfg = base_sim_cfg or SimulationConfig2D()

    def run_campaign(
        self,
        guidance_law: GuidanceLaw2D,
        scenario_id: str = "D",
        target_kwargs: Optional[Dict[str, Any]] = None,
        mc_config: Optional[MonteCarloConfig2D] = None,
    ) -> MonteCarloSummary2D:
        """
        Execute full Monte Carlo dispersion campaign across specified 2D guidance law and target.
        """
        cfg = mc_config or MonteCarloConfig2D()
        rng = np.random.default_rng(cfg.random_seed)

        run_results: List[MonteCarloRunResult2D] = []

        for i in range(cfg.num_runs):
            # 1. Perturb missile configuration
            v0_noise = rng.normal(0.0, cfg.sigma_v0_speed)
            v0_perturbed = max(100.0, self.base_missile_cfg.v0_magnitude + v0_noise)

            thrust_factor = 1.0 + rng.normal(0.0, cfg.sigma_thrust_pct)
            thrust_perturbed = self.base_missile_cfg.thrust_nominal * max(0.8, min(1.2, thrust_factor))

            m_cfg = replace(
                self.base_missile_cfg,
                v0_magnitude=v0_perturbed,
                thrust_nominal=thrust_perturbed,
            )

            # 2. Perturb target configuration
            speed_noise = rng.normal(0.0, cfg.sigma_target_speed)
            speed_perturbed = max(100.0, self.base_target_cfg.speed + speed_noise)

            pos_noise = rng.normal(0.0, cfg.sigma_target_r0, size=2)
            r0_perturbed = self.base_target_cfg.r0 + pos_noise

            t_cfg = replace(
                self.base_target_cfg,
                speed=speed_perturbed,
                r0=r0_perturbed,
            )

            # 3. Instantiate perturbed target
            kwargs = dict(target_kwargs or {})
            target = create_target_maneuver_2d(scenario_id, t_cfg, **kwargs)

            # 4. Run simulation
            engine = SimulationEngine2D(m_cfg, t_cfg, self.base_sim_cfg)
            res = engine.run(guidance_law, target)

            is_kill = res.miss_distance <= cfg.lethal_radius
            run_results.append(
                MonteCarloRunResult2D(
                    run_id=i + 1,
                    miss_distance=res.miss_distance,
                    intercept_time=res.intercept_time,
                    peak_lateral_g=res.peak_lateral_g,
                    total_control_energy=res.total_control_energy,
                    is_kill=is_kill,
                )
            )

        # Statistical analysis
        misses = np.array([r.miss_distance for r in run_results], dtype=np.float64)
        times = np.array([r.intercept_time for r in run_results], dtype=np.float64)
        energies = np.array([r.total_control_energy for r in run_results], dtype=np.float64)
        peak_gs = np.array([r.peak_lateral_g for r in run_results], dtype=np.float64)
        kills = np.array([r.is_kill for r in run_results], dtype=bool)

        pk = float(np.mean(kills))
        cep = float(np.percentile(misses, 50))
        r95 = float(np.percentile(misses, 95))
        mean_miss = float(np.mean(misses))
        std_miss = float(np.std(misses))
        median_miss = float(np.median(misses))
        mean_time = float(np.mean(times))
        mean_energy = float(np.mean(energies))
        mean_g = float(np.mean(peak_gs))

        return MonteCarloSummary2D(
            law=guidance_law,
            target_scenario=scenario_id,
            total_runs=cfg.num_runs,
            kill_probability_pk=pk,
            mean_miss_distance=mean_miss,
            std_miss_distance=std_miss,
            median_miss_distance=median_miss,
            cep_50=cep,
            r95=r95,
            mean_flight_time=mean_time,
            mean_control_energy=mean_energy,
            mean_peak_g=mean_g,
            runs=run_results,
        )
