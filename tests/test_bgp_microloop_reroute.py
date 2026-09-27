"""Unit tests for BGP Micro-Loop Elimination and TI-LFA Fast Reroute."""

import unittest
from datacenter_np_hard_kernel.core.bgp_microloop_reroute import BgpFastRerouteSolver
from datacenter_np_hard_kernel.core.models import FastRerouteResult, RouterLink


class TestBgpMicroloopReroute(unittest.TestCase):
    def setUp(self):
        self.solver = BgpFastRerouteSolver()

    def test_leaf_spine_topology_fast_reroute(self):
        # 2 spines, 3 leaves
        routers = ["s1", "s2", "l1", "l2", "l3"]
        links = [
            RouterLink("s1", "l1", 10),
            RouterLink("s1", "l2", 10),
            RouterLink("s1", "l3", 10),
            RouterLink("s2", "l1", 10),
            RouterLink("s2", "l2", 10),
            RouterLink("s2", "l3", 10),
            RouterLink("l1", "s1", 10),
            RouterLink("l1", "s2", 10),
            RouterLink("l2", "s1", 10),
            RouterLink("l2", "s2", 10),
            RouterLink("l3", "s1", 10),
            RouterLink("l3", "s2", 10),
        ]

        res = self.solver.solve(routers, links)
        self.assertIsInstance(res, FastRerouteResult)
        self.assertTrue(res.microloop_free_guarantee)
        self.assertGreaterEqual(res.coverage_percentage, 90.0)
        self.assertLessEqual(res.failover_latency_us, 10.0)
        self.assertIn("l1", res.primary_next_hops)


if __name__ == "__main__":
    unittest.main()
