"""Unit tests for Distributed AI All-Reduce and OCS Topology."""

import unittest
from datacenter_np_hard_kernel.core.collective_allreduce_topology import CollectiveTopologySolver
from datacenter_np_hard_kernel.core.models import CollectiveScheduleResult, GpuDevice


class TestCollectiveTopology(unittest.TestCase):
    def setUp(self):
        self.solver = CollectiveTopologySolver()

    def test_ring_vs_tree_scaling(self):
        # Ring latency scales O(N), Tree scales O(log N)
        t_ring_small = CollectiveTopologySolver.ring_allreduce_time(8, 100.0, 400.0)
        t_tree_small = CollectiveTopologySolver.double_binary_tree_time(8, 100.0, 400.0)

        t_ring_large = CollectiveTopologySolver.ring_allreduce_time(2048, 100.0, 400.0)
        t_tree_large = CollectiveTopologySolver.double_binary_tree_time(2048, 100.0, 400.0)

        # For very large clusters, Tree All-Reduce has much lower latency component than Ring
        self.assertGreater(t_ring_large, t_tree_large)

    def test_birkhoff_von_neumann_decomposition(self):
        matrix = [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
        configs = CollectiveTopologySolver.birkhoff_von_neumann_ocs(matrix)
        self.assertGreater(len(configs), 0)
        # Check permutations
        for weight, perm in configs:
            self.assertGreater(weight, 0.0)
            self.assertEqual(len(set(perm)), 3)

    def test_full_solver_run(self):
        gpus = [GpuDevice(f"gpu_{i}", f"host_{i // 8}", f"rack_{i // 32}") for i in range(128)]
        res = self.solver.solve(gpus, tensor_size_mb=256.0, topology_type="optical_circuit_switch")
        self.assertIsInstance(res, CollectiveScheduleResult)
        self.assertEqual(res.num_gpus, 128)
        self.assertGreater(res.theoretical_transfer_time_ms, 0.0)
        self.assertGreater(res.bus_bandwidth_efficiency, 0.0)
        self.assertGreater(res.model_flops_utilization_estimate, 0.0)


if __name__ == "__main__":
    unittest.main()
