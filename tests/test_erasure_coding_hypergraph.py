"""Unit tests for Exabyte Storage Local Reconstruction Codes (LRC)."""

import unittest
from datacenter_np_hard_kernel.core.erasure_coding_hypergraph import ErasureCodingHypergraphSolver
from datacenter_np_hard_kernel.core.models import ErasureLayoutResult


class TestErasureCodingHypergraph(unittest.TestCase):
    def setUp(self):
        self.solver = ErasureCodingHypergraphSolver(k_data=12, l_local=2, g_global=2)

    def test_chunk_generation_and_overhead(self):
        chunks = self.solver.generate_lrc_chunks("obj_123", size_mb=768.0)
        # Total chunks = k + l + g = 12 + 2 + 2 = 16
        self.assertEqual(len(chunks), 16)
        data_chunks = [c for c in chunks if c.chunk_type == "data"]
        local_parities = [c for c in chunks if c.chunk_type == "local_parity"]
        global_parities = [c for c in chunks if c.chunk_type == "global_parity"]

        self.assertEqual(len(data_chunks), 12)
        self.assertEqual(len(local_parities), 2)
        self.assertEqual(len(global_parities), 2)

    def test_hypergraph_placement_and_reconstruction_amplification(self):
        fault_domains = [
            {"node": f"node_{i}", "rack": f"rack_{i // 2}", "pdu": f"pdu_{i // 4}"}
            for i in range(16)
        ]
        res = self.solver.solve_placement("obj_123", fault_domains)
        self.assertIsInstance(res, ErasureLayoutResult)
        # 16 chunks / 12 data = 1.33x storage overhead
        self.assertAlmostEqual(res.storage_overhead, 1.333, places=2)
        # Repair amplification: 6 chunks read instead of 12 (50% reduction)
        self.assertEqual(res.reconstruction_read_amplification, 6.0)
        self.assertEqual(res.repair_io_reduction_pct, 50.0)
        self.assertGreater(res.fault_domains_covered, 1)


if __name__ == "__main__":
    unittest.main()
