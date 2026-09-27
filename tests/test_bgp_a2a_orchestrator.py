"""Unit tests for BGP in the Agentic AI Era and A2A Protocols."""

import unittest

from datacenter_np_hard_kernel.core.bgp_a2a_orchestrator import (
    BgpEgressPeerOptimizer,
    BgpEvpnRouteAggregator,
    BgpFlowspecVerifier,
    BgpRouteReflectorOptimizer,
    BgpStablePathsSolver,
)
from datacenter_np_hard_kernel.core.models import (
    A2AFlow,
    AsnPolicy,
    DisputeWheelResult,
    EpeAllocationResult,
    EpePeeringLink,
    EvpnAggregationResult,
    FlowspecRule,
    FlowspecVerificationResult,
    RouteReflectorResult,
)


class TestBgpStablePaths(unittest.TestCase):
    def setUp(self):
        self.solver = BgpStablePathsSolver()

    def test_acyclic_policy_is_stable(self):
        # Clean hierarchical policies: AS100 -> AS200 -> AS300
        policies = [
            AsnPolicy(100, "Enterprise-A", customers=[200], preferred_paths=[[100, 200]]),
            AsnPolicy(200, "Transit-B", providers=[100], customers=[300], preferred_paths=[[200, 300]]),
            AsnPolicy(300, "Agent-C", providers=[200], preferred_paths=[[300]]),
        ]
        res = self.solver.detect_and_resolve_dispute_wheel(policies)
        self.assertIsInstance(res, DisputeWheelResult)
        self.assertFalse(res.has_dispute_wheel)
        self.assertTrue(res.is_stable)
        self.assertEqual(len(res.cycles_detected), 0)

    def test_dispute_wheel_detection_and_tie_breaker(self):
        # Classic 3-node Dispute Wheel: AS10 prefers AS20, AS20 prefers AS30, AS30 prefers AS10
        policies = [
            AsnPolicy(10, "AS-10", preferred_paths=[[10, 20, 0]]),
            AsnPolicy(20, "AS-20", preferred_paths=[[20, 30, 0]]),
            AsnPolicy(30, "AS-30", preferred_paths=[[30, 10, 0]]),
        ]
        res = self.solver.detect_and_resolve_dispute_wheel(policies)
        self.assertTrue(res.has_dispute_wheel)
        self.assertEqual(len(res.cycles_detected), 1)
        self.assertEqual(len(res.cycles_detected[0]), 3)
        # Deterministic tie-breaker injected to eliminate oscillation
        self.assertIn(10, res.tie_breakers_injected)
        self.assertIn(20, res.tie_breakers_injected)
        self.assertIn(30, res.tie_breakers_injected)
        self.assertTrue(res.is_stable)


class TestBgpEgressPeerOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = BgpEgressPeerOptimizer(baseline_internet_rate_per_gb=0.08)

    def test_a2a_egress_unit_economics_and_sla(self):
        # 3 Peering options:
        # 1. Direct Connect: 15ms latency, $0.025/GB
        # 2. Equinix Fabric: 5ms latency, $0.003/GB
        # 3. Public Internet: 60ms latency, $0.080/GB
        links = [
            EpePeeringLink("link_dc", 65001, 16509, "direct_connect", cost_per_gb=0.025, latency_ms=15.0, capacity_gbps=100.0, srv6_sid="fc00:1::1"),
            EpePeeringLink("link_eqx", 65001, 24115, "equinix_fabric", cost_per_gb=0.003, latency_ms=5.0, capacity_gbps=400.0, srv6_sid="fc00:2::1"),
            EpePeeringLink("link_pub", 65001, 3356, "public_internet", cost_per_gb=0.080, latency_ms=60.0, capacity_gbps=100.0, srv6_sid="fc00:3::1"),
        ]

        flows = [
            # Latency-critical TTFT inference flow (100 GB)
            A2AFlow("flow_inference_ttft", "agent_root", "agent_worker_1", "aws_us_east", "azure_eastus2", volume_gb=100.0, is_latency_critical=True, max_latency_ms=20.0),
            # Bulk context dump (10,000 GB)
            A2AFlow("flow_context_dump", "agent_memory", "agent_rag", "aws_us_east", "coreweave_ord1", volume_gb=10000.0, is_latency_critical=False, max_latency_ms=100.0),
        ]

        res = self.optimizer.optimize_egress(flows, links)
        self.assertIsInstance(res, EpeAllocationResult)
        self.assertTrue(res.high_priority_latency_sla_met)
        # Latency-critical goes to direct connect / lowest valid latency
        self.assertIn("link_", res.allocations["flow_inference_ttft"])
        # Bulk context dump goes to Equinix Fabric ($0.003/GB)
        self.assertEqual(res.allocations["flow_context_dump"], "link_eqx")
        # Unit economics: Baseline is 10,100 GB * $0.08 = $808.00
        # Optimized is 100 * $0.003 (or $0.025) + 10,000 * $0.003 = ~$30.30 - $32.50
        self.assertGreater(res.cost_reduction_pct, 90.0)
        self.assertGreater(res.annual_savings_usd, 250000.0)


class TestBgpRouteReflectorOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = BgpRouteReflectorOptimizer(add_path_count=4)

    def test_route_reflector_placement_and_deflection(self):
        nodes = ["spine_1", "spine_2", "leaf_1", "leaf_2", "leaf_3", "leaf_4"]
        edges = [
            ("spine_1", "leaf_1", 10),
            ("spine_1", "leaf_2", 10),
            ("spine_1", "leaf_3", 10),
            ("spine_1", "leaf_4", 10),
            ("spine_2", "leaf_1", 10),
            ("spine_2", "leaf_2", 10),
            ("spine_2", "leaf_3", 10),
            ("spine_2", "leaf_4", 10),
        ]
        res = self.optimizer.optimize_placement(nodes, edges)
        self.assertIsInstance(res, RouteReflectorResult)
        self.assertTrue(res.deflection_free)
        # Spines should be chosen as RRs due to higher betweenness centrality
        self.assertIn("spine_1", res.selected_rr_nodes)
        self.assertEqual(res.add_path_count, 4)
        self.assertGreater(res.tcams_saved_pct, 0.0)


class TestBgpEvpnRouteAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = BgpEvpnRouteAggregator(fib_warning_threshold=50000)

    def test_prefix_compaction(self):
        # 16 contiguous agent IPs in 10.0.0.0/28
        host_ips = [f"10.0.0.{i}" for i in range(16)]
        res = self.aggregator.aggregate_routes(host_ips)
        self.assertIsInstance(res, EvpnAggregationResult)
        self.assertEqual(res.original_routes_count, 16)
        # 16 contiguous /32 host IPs collapse into a single /28 prefix
        self.assertEqual(res.compacted_prefixes_count, 1)
        self.assertAlmostEqual(res.compression_ratio_pct, 93.75, places=1)
        self.assertTrue(res.fib_overflow_prevented)


class TestBgpFlowspecVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = BgpFlowspecVerifier()

    def test_conflict_and_shadowing_detection(self):
        rules = [
            # High priority broad quarantine rule
            FlowspecRule(
                rule_id="r1_quarantine",
                agent_sentinel_id="soc_agent_01",
                src_prefix="10.0.0.0/16",
                dst_prefix="192.168.1.0/24",
                protocol=6,
                dst_port_range=(1000, 9000),
                action="drop",
                priority=200,
            ),
            # Low priority redundant shadowed rule with same action
            FlowspecRule(
                rule_id="r2_redundant",
                agent_sentinel_id="soc_agent_02",
                src_prefix="10.0.1.0/24",
                dst_prefix="192.168.1.50/32",
                protocol=6,
                dst_port_range=(8000, 8080),
                action="drop",
                priority=100,
            ),
            # Contradictory rule with opposite action ("rate_limit" inside dropped space)
            FlowspecRule(
                rule_id="r3_contradiction",
                agent_sentinel_id="soc_agent_03",
                src_prefix="10.0.2.0/24",
                dst_prefix="192.168.1.60/32",
                protocol=6,
                dst_port_range=(8000, 8080),
                action="rate_limit",
                priority=50,
            ),
        ]

        res = self.verifier.verify_rules(rules)
        self.assertIsInstance(res, FlowspecVerificationResult)
        # r2 should be shadowed by r1
        self.assertIn(("r2_redundant", "r1_quarantine"), res.shadowed_rules)
        # r3 should contradict r1
        self.assertIn(("r1_quarantine", "r3_contradiction"), res.contradictions)
        # Invalidation due to contradiction
        self.assertFalse(res.is_valid)


if __name__ == "__main__":
    unittest.main()
