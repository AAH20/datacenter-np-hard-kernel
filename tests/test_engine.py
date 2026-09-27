"""Unit tests for Unified Hyperscale Datacenter Engine."""

import unittest
from datacenter_np_hard_kernel.engine import HyperscaleBenchmarkReport, HyperscaleDatacenterEngine


class TestHyperscaleDatacenterEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HyperscaleDatacenterEngine()

    def test_run_full_benchmark(self):
        report = self.engine.run_full_benchmark()
        self.assertIsInstance(report, HyperscaleBenchmarkReport)

        # Verification of benchmark thresholds
        self.assertGreater(report.packing_utilization_pct, 40.0)
        self.assertGreater(report.wan_link_utilization_pct, 30.0)
        self.assertGreater(report.allreduce_bandwidth_efficiency, 0.0)
        self.assertGreater(report.carbon_reduction_pct, 0.0)
        self.assertEqual(report.erasure_repair_bandwidth_saved_pct, 50.0)
        self.assertLessEqual(report.bgp_failover_latency_us, 10.0)
        # Sub-second execution guarantee
        self.assertLess(report.total_pipeline_time_ms, 1000.0)


if __name__ == "__main__":
    unittest.main()
