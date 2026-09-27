"""Unit tests for Optical WAN Traffic Engineering."""

import unittest
from datacenter_np_hard_kernel.core.models import TrafficDemand, TrafficEngineeringResult, WanEdge, WanNode
from datacenter_np_hard_kernel.core.optical_traffic_engineering import OpticalTrafficEngineeringSolver


class TestOpticalTrafficEngineering(unittest.TestCase):
    def setUp(self):
        self.solver = OpticalTrafficEngineeringSolver(target_max_utilization=0.95)

    def test_single_flow_within_capacity(self):
        nodes = [WanNode("dc1", "us-east", "DC1"), WanNode("dc2", "us-west", "DC2")]
        edges = [WanEdge("e1", "dc1", "dc2", capacity_gbps=100.0, latency_ms=40.0)]
        demands = [TrafficDemand("flow1", "dc1", "dc2", volume_gbps=50.0, priority="interactive")]

        res = self.solver.solve(nodes, edges, demands)
        self.assertIsInstance(res, TrafficEngineeringResult)
        self.assertEqual(res.satisfied_demand_gbps, 50.0)
        self.assertEqual(res.high_priority_loss_rate, 0.0)
        self.assertAlmostEqual(res.link_utilizations["e1"], 0.5)

    def test_priority_scheduling_and_multipath(self):
        nodes = [
            WanNode("dc1", "us-east", "DC1"),
            WanNode("dc2", "us-west", "DC2"),
            WanNode("dc3", "central", "DC3"),
        ]
        edges = [
            WanEdge("e12", "dc1", "dc2", capacity_gbps=100.0, latency_ms=50.0),
            WanEdge("e13", "dc1", "dc3", capacity_gbps=80.0, latency_ms=25.0),
            WanEdge("e32", "dc3", "dc2", capacity_gbps=80.0, latency_ms=25.0),
        ]
        # Demands exceed direct link capacity
        demands = [
            TrafficDemand("user_rpc", "dc1", "dc2", volume_gbps=80.0, priority="interactive", max_latency_ms=60.0),
            TrafficDemand("backup_sync", "dc1", "dc2", volume_gbps=60.0, priority="batch", max_latency_ms=100.0),
        ]

        res = self.solver.solve(nodes, edges, demands)
        self.assertGreater(res.satisfied_demand_gbps, 100.0)
        self.assertEqual(res.high_priority_loss_rate, 0.0)
        self.assertLessEqual(res.max_link_utilization, 0.96)


if __name__ == "__main__":
    unittest.main()
