"""Hyperscale Distributed Data Center & Cloud NP-Hard Combinatorial Kernel.

Exports the 6 apex NP-Hard and APX-Hard solvers for hyperscale clouds:
- VectorBinPackingSolver (Compute / GPU Gang Scheduling)
- OpticalTrafficEngineeringSolver (Constrained Multi-Commodity Flow WAN)
- CollectiveTopologySolver (Distributed AI All-Reduce & OCS Synthesis)
- PowerThermalScheduler (Carbon-Aware & Thermal Dynamic Power)
- ErasureCodingHypergraphSolver (Local Reconstruction Codes & Fault Domains)
- BgpFastRerouteSolver (TI-LFA Micro-Loop Elimination)
- HyperscaleDatacenterEngine (Unified Orchestrator)
"""

from .core import (
    BgpFastRerouteSolver,
    CollectiveScheduleResult,
    CollectiveTopologySolver,
    ComputeJob,
    ErasureCodingHypergraphSolver,
    ErasureLayoutResult,
    FastRerouteResult,
    GangJob,
    GpuDevice,
    GridCarbonSlot,
    HostNode,
    OpticalTrafficEngineeringSolver,
    PackingResult,
    PowerScheduleResult,
    PowerThermalScheduler,
    RouterLink,
    StorageChunk,
    TaskPod,
    TrafficDemand,
    TrafficEngineeringResult,
    VectorBinPackingSolver,
    WanEdge,
    WanNode,
    AsnPolicy,
    DisputeWheelResult,
    A2AFlow,
    EpePeeringLink,
    EpeAllocationResult,
    RouteReflectorResult,
    EvpnAggregationResult,
    FlowspecRule,
    FlowspecVerificationResult,
    BgpStablePathsSolver,
    BgpEgressPeerOptimizer,
    BgpRouteReflectorOptimizer,
    BgpEvpnRouteAggregator,
    BgpFlowspecVerifier,
)
from .engine import HyperscaleBenchmarkReport, HyperscaleDatacenterEngine

__version__ = "0.2.0"

__all__ = [
    # Engine
    "HyperscaleDatacenterEngine",
    "HyperscaleBenchmarkReport",
    # Core Solvers
    "VectorBinPackingSolver",
    "OpticalTrafficEngineeringSolver",
    "CollectiveTopologySolver",
    "PowerThermalScheduler",
    "ErasureCodingHypergraphSolver",
    "BgpFastRerouteSolver",
    # BGP in Agentic AI & A2A Solvers
    "BgpStablePathsSolver",
    "BgpEgressPeerOptimizer",
    "BgpRouteReflectorOptimizer",
    "BgpEvpnRouteAggregator",
    "BgpFlowspecVerifier",
    # Models
    "HostNode",
    "TaskPod",
    "GangJob",
    "PackingResult",
    "WanNode",
    "WanEdge",
    "TrafficDemand",
    "TrafficEngineeringResult",
    "GpuDevice",
    "CollectiveScheduleResult",
    "GridCarbonSlot",
    "ComputeJob",
    "PowerScheduleResult",
    "StorageChunk",
    "ErasureLayoutResult",
    "RouterLink",
    "FastRerouteResult",
    "AsnPolicy",
    "DisputeWheelResult",
    "A2AFlow",
    "EpePeeringLink",
    "EpeAllocationResult",
    "RouteReflectorResult",
    "EvpnAggregationResult",
    "FlowspecRule",
    "FlowspecVerificationResult",
]

