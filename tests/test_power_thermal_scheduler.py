"""Unit tests for Power, Thermal, and Carbon-Aware Job Scheduling."""

import unittest
from datacenter_np_hard_kernel.core.models import ComputeJob, GridCarbonSlot, PowerScheduleResult
from datacenter_np_hard_kernel.core.power_thermal_scheduler import PowerThermalScheduler


class TestPowerThermalScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = PowerThermalScheduler(peak_substation_cap_mw=100.0)

    def test_carbon_aware_spatial_and_temporal_shifting(self):
        # Two regions: Region A has solar peak at slot 4 (low carbon), Region B is dirty coal
        slots = [
            GridCarbonSlot("region_a", 0, carbon_g_per_kwh=400.0, renewable_fraction=0.1, max_power_mw=50.0),
            GridCarbonSlot("region_a", 1, carbon_g_per_kwh=350.0, renewable_fraction=0.2, max_power_mw=50.0),
            GridCarbonSlot("region_a", 2, carbon_g_per_kwh=50.0, renewable_fraction=0.9, max_power_mw=50.0),  # Green Peak
            GridCarbonSlot("region_a", 3, carbon_g_per_kwh=60.0, renewable_fraction=0.85, max_power_mw=50.0), # Green Peak
            GridCarbonSlot("region_b", 0, carbon_g_per_kwh=600.0, renewable_fraction=0.05, max_power_mw=50.0),
            GridCarbonSlot("region_b", 1, carbon_g_per_kwh=600.0, renewable_fraction=0.05, max_power_mw=50.0),
            GridCarbonSlot("region_b", 2, carbon_g_per_kwh=600.0, renewable_fraction=0.05, max_power_mw=50.0),
            GridCarbonSlot("region_b", 3, carbon_g_per_kwh=600.0, renewable_fraction=0.05, max_power_mw=50.0),
        ]
        # Job can be scheduled anytime before slot 4
        jobs = [
            ComputeJob("batch_ml_train", power_mw=20.0, duration_slots=2, deadline_slot=4, is_flexible=True),
        ]

        res = self.scheduler.solve(slots, jobs)
        self.assertIsInstance(res, PowerScheduleResult)
        # Job should be placed in region_a starting at slot 2 (green peak)
        region, start, end = res.job_placements["batch_ml_train"]
        self.assertEqual(region, "region_a")
        self.assertEqual(start, 2)
        self.assertEqual(end, 4)
        self.assertGreater(res.carbon_reduction_pct, 0.0)
        self.assertGreater(res.average_renewable_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
