"""Data models for Hyperscale Distributed Data Center and Cloud NP-Hard Solvers.

Encapsulates data structures for:
1. Multi-Dimensional Vector Bin Packing & Gang Scheduling
2. Constrained Optical WAN Traffic Engineering
3. Distributed AI All-Reduce & Optical Circuit Switch (OCS) Topology
4. Non-Linear Power, Thermal, & Carbon-Aware Job Scheduling
5. Exabyte Storage Local Reconstruction Codes & Hypergraph Placement
6. BGP Micro-Loop Free Alternate Fast Rerouting
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Vector Bin Packing & Gang Scheduling Models
# ============================================================================

@dataclasses.dataclass
class HostNode:
    """A physical hyperscale server with multi-dimensional resource capacities."""
    node_id: str
    rack_id: str
    cluster_id: str
    capacities: Dict[str, float]  # cpu, ram_gb, gpus, gpu_mem_gb, nvlink_bw, power_w
    allocated: Dict[str, float] = dataclasses.field(default_factory=dict)
    numa_domains: int = 2
    tags: Set[str] = dataclasses.field(default_factory=set)

    def remaining_capacity(self) -> Dict[str, float]:
        res = {}
        for k, cap in self.capacities.items():
            used = self.allocated.get(k, 0.0)
            res[k] = max(0.0, cap - used)
        return res

    def can_fit(self, demands: Dict[str, float]) -> bool:
        rem = self.remaining_capacity()
        for k, dem in demands.items():
            if rem.get(k, 0.0) < dem:
                return False
        return True

    def allocate(self, demands: Dict[str, float]) -> None:
        for k, dem in demands.items():
            self.allocated[k] = self.allocated.get(k, 0.0) + dem

    def deallocate(self, demands: Dict[str, float]) -> None:
        for k, dem in demands.items():
            self.allocated[k] = max(0.0, self.allocated.get(k, 0.0) - dem)


@dataclasses.dataclass
class TaskPod:
    """A container, VM, or GPU worker task demand."""
    pod_id: str
    job_id: str
    demands: Dict[str, float]
    gang_id: Optional[str] = None
    affinity_tags: Set[str] = dataclasses.field(default_factory=set)
    anti_affinity_tags: Set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class GangJob:
    """An all-or-nothing multi-pod distributed training or HPC workload."""
    job_id: str
    pods: List[TaskPod]
    require_same_rack: bool = False
    max_switch_hops: int = 2


@dataclasses.dataclass
class PackingResult:
    """Outcome of multi-dimensional vector bin packing and gang scheduling."""
    assignments: Dict[str, str]           # pod_id -> node_id
    unassigned_pods: List[str]
    node_utilization: Dict[str, float]    # node_id -> average resource utilization in [0, 1]
    stranded_resources: Dict[str, float]  # resource_name -> percentage of stranded capacity
    gang_success: bool
    solve_time_ms: float
    total_active_nodes: int


# ============================================================================
# 2. Optical WAN Traffic Engineering Models
# ============================================================================

@dataclasses.dataclass
class WanNode:
    """A Point of Presence (PoP) or Regional Data Center gateway."""
    node_id: str
    region: str
    datacenter_name: str


@dataclasses.dataclass
class WanEdge:
    """A fiber-optic trunk link connecting data centers."""
    edge_id: str
    src: str
    dst: str
    capacity_gbps: float
    latency_ms: float
    cost: float = 1.0


@dataclasses.dataclass
class TrafficDemand:
    """A bandwidth demand between two cloud regions."""
    flow_id: str
    src: str
    dst: str
    volume_gbps: float
    priority: str = "batch"  # "interactive" or "batch"
    max_latency_ms: float = 150.0


@dataclasses.dataclass
class TrafficEngineeringResult:
    """Outcome of Multi-Commodity Flow WAN traffic engineering."""
    allocations: Dict[str, List[Tuple[List[str], float]]]  # flow_id -> list of (path, bandwidth_gbps)
    satisfied_demand_gbps: float
    total_demand_gbps: float
    link_utilizations: Dict[str, float]                    # edge_id -> fraction utilized
    average_utilization: float
    max_link_utilization: float
    high_priority_loss_rate: float
    solve_time_ms: float


# ============================================================================
# 3. Distributed AI Collective All-Reduce & OCS Models
# ============================================================================

@dataclasses.dataclass
class GpuDevice:
    """An accelerator participating in distributed collective communication."""
    gpu_id: str
    host_id: str
    rack_id: str
    intra_host_bandwidth_gbps: float = 900.0   # NVLink 4 / 5
    inter_host_bandwidth_gbps: float = 400.0   # RoCEv2 / InfiniBand NDR


@dataclasses.dataclass
class CollectiveScheduleResult:
    """Outcome of collective communication scheduling and optical reconfiguration."""
    algorithm: str
    num_gpus: int
    tensor_size_mb: float
    theoretical_transfer_time_ms: float
    bus_bandwidth_efficiency: float
    model_flops_utilization_estimate: float
    optical_reconfigurations_used: int
    straggler_overhead_ms: float


# ============================================================================
# 4. Power, Thermal & Carbon-Aware Job Scheduling Models
# ============================================================================

@dataclasses.dataclass
class GridCarbonSlot:
    """Regional power grid carbon intensity and renewable supply at time t."""
    region: str
    time_slot: int
    carbon_g_per_kwh: float
    renewable_fraction: float
    max_power_mw: float


@dataclasses.dataclass
class ComputeJob:
    """Workload with power demand and temporal/spatial flexibility."""
    job_id: str
    power_mw: float
    duration_slots: int
    deadline_slot: int
    is_flexible: bool = True
    admissible_regions: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PowerScheduleResult:
    """Outcome of carbon-intelligent spatial and temporal job scheduling."""
    job_placements: Dict[str, Tuple[str, int, int]]  # job_id -> (region, start_slot, end_slot)
    total_carbon_kg: float
    carbon_reduction_pct: float
    peak_power_shaved_mw: float
    average_renewable_fraction: float
    thermal_throttling_avoided_events: int


# ============================================================================
# 5. Exabyte Storage Local Reconstruction Codes (LRC) Models
# ============================================================================

@dataclasses.dataclass
class StorageChunk:
    """Fragment of an erasure-coded object."""
    chunk_id: str
    object_id: str
    chunk_type: str   # "data", "local_parity", "global_parity"
    group_id: int
    size_mb: float = 64.0


@dataclasses.dataclass
class ErasureLayoutResult:
    """Outcome of LRC layout and hypergraph fault domain placement."""
    code_profile: str  # e.g., "LRC(12, 2, 2)"
    storage_overhead: float
    reconstruction_read_amplification: float
    chunk_to_node_map: Dict[str, str]
    fault_domains_covered: int
    tolerated_arbitrary_failures: int
    repair_io_reduction_pct: float


# ============================================================================
# 6. BGP Micro-Loop Elimination & Fast Reroute Models
# ============================================================================

@dataclasses.dataclass
class RouterLink:
    src: str
    dst: str
    metric: int
    is_alive: bool = True


@dataclasses.dataclass
class FastRerouteResult:
    """Outcome of TI-LFA backup path calculation and micro-loop elimination."""
    primary_next_hops: Dict[str, Dict[str, str]]    # (src, dst) -> next_hop
    backup_ti_lfa_paths: Dict[str, Dict[str, List[str]]]  # (failed_link, dst) -> repair_path
    microloop_free_guarantee: bool
    coverage_percentage: float
    failover_latency_us: float
