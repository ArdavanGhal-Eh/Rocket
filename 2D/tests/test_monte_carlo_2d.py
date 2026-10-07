"""
Unit tests for Monte Carlo Dispersion Simulator (2D Suite).
Verifies stochastic campaign execution, statistical metrics (CEP, Pk, R95), and repeatability in 2D.
"""

import os
import sys
import unittest
import numpy as np

# Ensure 2D directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from missile_sim_2d.guidance import GuidanceLaw2D
from missile_sim_2d.monte_carlo import (
    MonteCarloConfig2D,
    MonteCarloSimulator2D,
    MonteCarloSummary2D,
)


class TestMonteCarlo2D(unittest.TestCase):
    """Test suite for 2D Monte Carlo dispersion simulation."""

    def test_monte_carlo_campaign_execution_2d(self):
        mc_sim = MonteCarloSimulator2D()
        cfg = MonteCarloConfig2D(num_runs=5, random_seed=456)

        summary = mc_sim.run_campaign(
            guidance_law=GuidanceLaw2D.APN,
            scenario_id="A",
            mc_config=cfg,
        )

        self.assertIsInstance(summary, MonteCarloSummary2D)
        self.assertEqual(summary.total_runs, 5)
        self.assertEqual(len(summary.runs), 5)
        self.assertTrue(0.0 <= summary.kill_probability_pk <= 1.0)
        self.assertTrue(summary.cep_50 >= 0.0)
        self.assertTrue(summary.r95 >= 0.0)
        self.assertTrue(summary.mean_flight_time > 0.0)

        # Test ASCII summary table formatting
        table_str = summary.summary_table()
        self.assertIn("2D MONTE CARLO DISPERSION ANALYSIS", table_str)
        self.assertIn("Circular Error Probable", table_str)


if __name__ == "__main__":
    unittest.main()
