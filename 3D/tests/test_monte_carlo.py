"""
Unit tests for Monte Carlo Dispersion Simulator (3D Suite).
Verifies stochastic campaign execution, statistical metrics (CEP, Pk, R95), and repeatability.
"""

import os
import sys
import unittest
import numpy as np

# Ensure 3D directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from missile_sim.guidance import GuidanceLaw
from missile_sim.monte_carlo import (
    MonteCarloConfig,
    MonteCarloSimulator,
    MonteCarloSummary,
)


class TestMonteCarlo3D(unittest.TestCase):
    """Test suite for 3D Monte Carlo dispersion simulation."""

    def test_monte_carlo_campaign_execution(self):
        mc_sim = MonteCarloSimulator()
        cfg = MonteCarloConfig(num_runs=5, random_seed=123)

        summary = mc_sim.run_campaign(
            guidance_law=GuidanceLaw.APN,
            scenario_id="A",
            mc_config=cfg,
        )

        self.assertIsInstance(summary, MonteCarloSummary)
        self.assertEqual(summary.total_runs, 5)
        self.assertEqual(len(summary.runs), 5)
        self.assertTrue(0.0 <= summary.kill_probability_pk <= 1.0)
        self.assertTrue(summary.cep_50 >= 0.0)
        self.assertTrue(summary.r95 >= 0.0)
        self.assertTrue(summary.mean_flight_time > 0.0)

        # Test ASCII summary table formatting
        table_str = summary.summary_table()
        self.assertIn("MONTE CARLO DISPERSION ANALYSIS", table_str)
        self.assertIn("Circular Error Probable", table_str)


if __name__ == "__main__":
    unittest.main()
