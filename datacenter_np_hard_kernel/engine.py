"""Hyperscale Distributed Data Center & Cloud NP-Hard Optimization Engine.

Provides unified orchestration across all 6 apex combinatorial solvers:
1. Multi-Dimensional Vector Bin Packing & Gang Scheduling
2. Optical WAN Multi-Commodity Flow Traffic Engineering
3. Distributed AI All-Reduce & OCS Topology Synthesis
4. Power, Thermal & Carbon-Aware Job Scheduling
5. Exabyte Storage Local Reconstruction Codes (LRC)
6. BGP Micro-Loop Elimination (TI-LFA)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional

from .core.bgp_microloop_reroute import BgpFastRerouteSolver
from .core.collective_allreduce_topology import CollectiveTopologySolver
from .core.erasure_coding_hypergraph import ErasureCodingHypergraphSolver
from .core.models import (
    ComputeJob,
    GpuDevice,
    GridCarbonSlot,
    HostNode,
    RouterLink,
    TaskPod,
    TrafficDemand,
    WanEdge,
    WanNode,
)
from .core.optical_traffic_engineering import OpticalTrafficEngineeringSolver
from .core.power_thermal_scheduler import PowerThermalScheduler
from .core.vector_bin_packing import VectorBinPackingSolver


@dataclasses.dataclass
class HyperscaleBenchmarkReport:
    """Summary of performance metrics and benchmarks across all 6 solvers."""
    packing_utilization_pct: float
    packing_solve_time_ms: float
    wan_link_utilization_pct: float
    wan_solve_time_ms: float
    allreduce_bandwidth_efficiency: float
    allreduce_mfu_estimate: float
    carbon_reduction_pct: float
    peak_power_shaved_mw: float
    erasure_repair_bandwidth_saved_pct: float
    bgp_failover_latency_us: float
    bgp_coverage_pct: float
    total_pipeline_time_ms: float


class HyperscaleDatacenterEngine:
    """End-to-End Orchestrator for Hyperscale Infrastructure Solvers."""

    def __init__(self):
        self.packing_solver = VectorBinPackingSolver()
        self.wan_solver = OpticalTrafficEngineeringSolver()
        self.collective_solver = CollectiveTopologySolver()
        self.power_scheduler = PowerThermalScheduler()
        self.erasure_solver = ErasureCodingHypergraphSolver()
        self.bgp_solver = BgpFastRerouteSolver()

    def run_full_benchmark(self) -> HyperscaleBenchmarkReport:
        """Executes a full multi-tier benchmark across all 6 optimization solvers."""
        t0 = time.perf_counter()

        # 1. Compute & GPU Bin Packing Benchmark
        nodes = [
            HostNode(
                node_id=f"node_{i:03d}",
                rack_id=f"rack_{i // 16:02d}",
                cluster_id="us-east-ai-cluster",
                capacities={"cpu": 128.0, "ram_gb": 1024.0, "gpus": 8.0, "power_w": 10200.0},
                tags={"gpu_h100", "nvlink4"} if i % 2 == 0 else {"cpu_general"},
            )
            for i in range(64)
        ]
        pods = [
            TaskPod(
                pod_id=f"pod_{j:04d}",
                job_id=f"job_{j // 8}",
                demands={"cpu": 16.0, "ram_gb": 128.0, "gpus": 1.0, "power_w": 1200.0},
                affinity_tags={"gpu_h100"} if j % 2 == 0 else set(),
            )
            for j in range(256)
        ]
        pack_res = self.packing_solver.solve_pods(nodes, pods)

        # 2. Optical WAN Traffic Engineering Benchmark
        wan_nodes = [
            WanNode(f"pop_{r}", r, f"DC-{r.upper()}")
            for r in ["us-east", "us-west", "eu-central", "ap-southeast"]
        ]
        wan_edges = [
            WanEdge("e1", "pop_us-east", "pop_us-west", capacity_gbps=1000.0, latency_ms=45.0),
            WanEdge("e2", "pop_us-east", "pop_eu-central", capacity_gbps=800.0, latency_ms=75.0),
            WanEdge("e3", "pop_eu-central", "pop_ap-southeast", capacity_gbps=600.0, latency_ms=120.0),
            WanEdge("e4", "pop_us-west", "pop_ap-southeast", capacity_gbps=600.0, latency_ms=110.0),
        ]
        demands = [
            TrafficDemand("d1", "pop_us-east", "pop_eu-central", volume_gbps=750.0, priority="interactive", max_latency_ms=80.0),
            TrafficDemand("d2", "pop_us-west", "pop_ap-southeast", volume_gbps=500.0, priority="batch", max_latency_ms=150.0),
        ]
        wan_res = self.wan_solver.solve(wan_nodes, wan_edges, demands)

        # 3. Distributed AI All-Reduce Benchmark
        gpus = [
            GpuDevice(f"gpu_{g:04d}", f"host_{g // 8:03d}", f"rack_{g // 32:02d}")
            for g in range(1024)
        ]
        coll_res = self.collective_solver.solve(gpus, tensor_size_mb=1024.0, topology_type="optical_circuit_switch")

        # 4. Power & Carbon Scheduler Benchmark
        grid_slots = [
            GridCarbonSlot("us-east", slot, carbon_g_per_kwh=420.0 - slot * 15.0, renewable_fraction=0.3 + slot * 0.03, max_power_mw=150.0)
            for slot in range(24)
        ] + [
            GridCarbonSlot("eu-north", slot, carbon_g_per_kwh=80.0 + slot * 5.0, renewable_fraction=0.85 - slot * 0.01, max_power_mw=120.0)
            for slot in range(24)
        ]
        jobs = [
            ComputeJob(f"batch_job_{k}", power_mw=10.0 + (k % 5) * 2.0, duration_slots=4, deadline_slot=22, is_flexible=True)
            for k in range(12)
        ]
        power_res = self.power_scheduler.solve(grid_slots, jobs)

        # 5. Storage Erasure Coding Benchmark
        fault_domains = [
            {"node": f"stor_node_{i:02d}", "rack": f"rack_{i // 4:02d}", "pdu": f"pdu_{i // 8:02d}"}
            for i in range(32)
        ]
        lrc_res = self.erasure_solver.solve_placement("obj_model_weights_v1", fault_domains)

        # 6. BGP Micro-Loop Elimination Benchmark
        routers = ["spine_1", "spine_2", "leaf_1", "leaf_2", "leaf_3", "leaf_4"]
        links = [
            RouterLink("spine_1", "leaf_1", 10),
            RouterLink("spine_1", "leaf_2", 10),
            RouterLink("spine_1", "leaf_3", 10),
            RouterLink("spine_1", "leaf_4", 10),
            RouterLink("spine_2", "leaf_1", 10),
            RouterLink("spine_2", "leaf_2", 10),
            RouterLink("spine_2", "leaf_3", 10),
            RouterLink("spine_2", "leaf_4", 10),
            RouterLink("leaf_1", "spine_1", 10),
            RouterLink("leaf_2", "spine_1", 10),
            RouterLink("leaf_3", "spine_1", 10),
            RouterLink("leaf_4", "spine_1", 10),
            RouterLink("leaf_1", "spine_2", 10),
            RouterLink("leaf_2", "spine_2", 10),
            RouterLink("leaf_3", "spine_2", 10),
            RouterLink("leaf_4", "spine_2", 10),
        ]
        bgp_res = self.bgp_solver.solve(routers, links)

        total_pipeline_time_ms = (time.perf_counter() - t0) * 1000.0

        avg_pack_util = sum(pack_res.node_utilization.values()) / max(1, len(pack_res.node_utilization))

        return HyperscaleBenchmarkReport(
            packing_utilization_pct=round(avg_pack_util * 100.0, 1),
            packing_solve_time_ms=round(pack_res.solve_time_ms, 2),
            wan_link_utilization_pct=round(wan_res.average_utilization * 100.0, 1),
            wan_solve_time_ms=round(wan_res.solve_time_ms, 2),
            allreduce_bandwidth_efficiency=coll_res.bus_bandwidth_efficiency,
            allreduce_mfu_estimate=coll_res.model_flops_utilization_estimate,
            carbon_reduction_pct=power_res.carbon_reduction_pct,
            peak_power_shaved_mw=power_res.peak_power_shaved_mw,
            erasure_repair_bandwidth_saved_pct=lrc_res.repair_io_reduction_pct,
            bgp_failover_latency_us=bgp_res.failover_latency_us,
            bgp_coverage_pct=bgp_res.coverage_percentage,
            total_pipeline_time_ms=round(total_pipeline_time_ms, 2),
        )
