"""Unit tests for Multi-Dimensional Vector Bin Packing & Gang Scheduling."""

import unittest
from datacenter_np_hard_kernel.core.models import GangJob, HostNode, PackingResult, TaskPod
from datacenter_np_hard_kernel.core.vector_bin_packing import (
    VectorBinPackingSolver,
    cosine_alignment,
    vector_dot,
    vector_norm,
)


class TestVectorBinPacking(unittest.TestCase):
    def setUp(self):
        self.solver = VectorBinPackingSolver()

    def test_vector_math(self):
        v1 = {"cpu": 3.0, "ram": 4.0}
        self.assertAlmostEqual(vector_norm(v1), 5.0)

        v2 = {"cpu": 3.0, "ram": 4.0}
        self.assertAlmostEqual(vector_dot(v1, v2), 25.0)
        self.assertAlmostEqual(cosine_alignment(v1, v2), 1.0)

        v_orth = {"gpu": 10.0}
        self.assertAlmostEqual(cosine_alignment(v1, v_orth), 0.0)

    def test_packing_individual_pods(self):
        nodes = [
            HostNode("node_1", "r1", "c1", {"cpu": 16.0, "ram_gb": 64.0}),
            HostNode("node_2", "r1", "c1", {"cpu": 16.0, "ram_gb": 64.0}),
        ]
        pods = [
            TaskPod("pod_1", "j1", {"cpu": 8.0, "ram_gb": 32.0}),
            TaskPod("pod_2", "j1", {"cpu": 8.0, "ram_gb": 32.0}),
            TaskPod("pod_3", "j2", {"cpu": 8.0, "ram_gb": 32.0}),
        ]

        result = self.solver.solve_pods(nodes, pods)
        self.assertIsInstance(result, PackingResult)
        self.assertEqual(len(result.assignments), 3)
        self.assertEqual(len(result.unassigned_pods), 0)
        self.assertTrue(result.gang_success)
        self.assertGreater(result.total_active_nodes, 0)

    def test_gang_scheduling_atomic_rollback(self):
        nodes = [
            HostNode("node_1", "r1", "c1", {"gpus": 4.0, "cpu": 32.0}),
        ]
        # Gang requires 8 GPUs total across 2 pods, but only 4 GPUs exist
        gang = GangJob(
            job_id="gang_llm",
            pods=[
                TaskPod("pod_a", "gang_llm", {"gpus": 4.0, "cpu": 16.0}),
                TaskPod("pod_b", "gang_llm", {"gpus": 4.0, "cpu": 16.0}),
            ],
        )

        result = self.solver.solve_gang(nodes, gang)
        # Gang must fail and atomically roll back
        self.assertFalse(result.gang_success)
        self.assertEqual(len(result.assignments), 0)
        self.assertEqual(len(result.unassigned_pods), 2)
        # Node capacity must remain 100% free
        self.assertEqual(nodes[0].allocated.get("gpus", 0.0), 0.0)

    def test_affinity_tags(self):
        nodes = [
            HostNode("gpu_node", "r1", "c1", {"cpu": 16.0}, tags={"gpu_h100"}),
            HostNode("cpu_node", "r1", "c1", {"cpu": 16.0}, tags={"cpu_only"}),
        ]
        pod = TaskPod("pod_ml", "job_1", {"cpu": 4.0}, affinity_tags={"gpu_h100"})

        result = self.solver.solve_pods(nodes, [pod])
        self.assertEqual(result.assignments["pod_ml"], "gpu_node")


if __name__ == "__main__":
    unittest.main()
