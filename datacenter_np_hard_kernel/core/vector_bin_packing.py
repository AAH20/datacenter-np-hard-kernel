"""Multi-Dimensional Vector Bin Packing & Gang Scheduling for Hyperscale GPU/CPU Clusters.

Addresses the APX-Hard problem of packing heterogeneous VMs, containers, and GPU
training jobs across multi-dimensional physical constraints (CPU, RAM, GPUs,
HBM, NVLink, Thermal TDP, and Network Egress) while preventing resource stranding.

Implements:
1. Tetris Vector Cosine Alignment: Aligns demand vectors with remaining capacity vectors.
2. All-or-Nothing Gang Scheduling: Atomically provisions multi-node training clusters.
3. Stranded Resource Minimization: Symmetrically exhausts multidimensional capacities.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

from .models import GangJob, HostNode, PackingResult, TaskPod


def vector_norm(v: Dict[str, float]) -> float:
    return math.sqrt(sum(val * val for val in v.values()))


def vector_dot(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    return sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in set(v1) | set(v2))


def cosine_alignment(demand: Dict[str, float], remaining: Dict[str, float]) -> float:
    """Computes cosine similarity between resource demand and remaining capacity."""
    norm_d = vector_norm(demand)
    norm_r = vector_norm(remaining)
    if norm_d <= 1e-9 or norm_r <= 1e-9:
        return 0.0
    dot = vector_dot(demand, remaining)
    return dot / (norm_d * norm_r)


class VectorBinPackingSolver:
    """Solves Multi-Dimensional Vector Bin Packing (MDVBP) with Gang Constraints."""

    def __init__(self, weight_alignment: float = 0.7, weight_utilization: float = 0.3):
        self.w_align = weight_alignment
        self.w_util = weight_utilization

    def score_node(self, node: HostNode, pod: TaskPod) -> float:
        """Scores candidate node using vector cosine alignment and residual density."""
        if not node.can_fit(pod.demands):
            return -1.0

        # Affinity check
        if pod.affinity_tags and not (pod.affinity_tags & node.tags):
            return -1.0
        if pod.anti_affinity_tags and (pod.anti_affinity_tags & node.tags):
            return -1.0

        rem = node.remaining_capacity()
        alignment = cosine_alignment(pod.demands, rem)

        # Average utilization after hypothetical allocation
        total_caps = sum(node.capacities.values())
        if total_caps <= 1e-9:
            return 0.0
        new_alloc = sum(node.allocated.get(k, 0.0) + pod.demands.get(k, 0.0) for k in node.capacities)
        util_score = new_alloc / total_caps

        return (self.w_align * alignment) + (self.w_util * util_score)

    def solve_pods(
        self,
        nodes: List[HostNode],
        pods: List[TaskPod],
    ) -> PackingResult:
        """Packs individual pods using Best-Fit Decreasing Vector Cosine heuristic."""
        start_time = time.perf_counter()
        node_map = {n.node_id: n for n in nodes}
        assignments: Dict[str, str] = {}
        unassigned: List[str] = []

        # Sort pods by multi-dimensional Euclidean magnitude descending
        sorted_pods = sorted(pods, key=lambda p: vector_norm(p.demands), reverse=True)

        for pod in sorted_pods:
            best_node_id: Optional[str] = None
            best_score = -1.0

            for node in nodes:
                score = self.score_node(node, pod)
                if score > best_score:
                    best_score = score
                    best_node_id = node.node_id

            if best_node_id is not None:
                assignments[pod.pod_id] = best_node_id
                node_map[best_node_id].allocate(pod.demands)
            else:
                unassigned.append(pod.pod_id)

        # Compute utilization and stranded capacity metrics
        node_utils, stranded = self._compute_metrics(nodes)
        active_nodes = sum(1 for n in nodes if any(n.allocated.values()))

        solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        return PackingResult(
            assignments=assignments,
            unassigned_pods=unassigned,
            node_utilization=node_utils,
            stranded_resources=stranded,
            gang_success=len(unassigned) == 0,
            solve_time_ms=solve_time_ms,
            total_active_nodes=active_nodes,
        )

    def solve_gang(
        self,
        nodes: List[HostNode],
        gang_job: GangJob,
    ) -> PackingResult:
        """Atomically provisions all pods in a GangJob or rolls back cleanly."""
        start_time = time.perf_counter()
        node_map = {n.node_id: n for n in nodes}
        trial_assignments: Dict[str, str] = {}
        rollback_log: List[Tuple[str, Dict[str, float]]] = []
        gang_success = True

        # Group candidate nodes by rack if rack-locality is enforced
        candidate_nodes = list(nodes)
        if gang_job.require_same_rack:
            # Pick rack with highest combined GPU/Compute capacity
            rack_capacities: Dict[str, float] = {}
            for n in nodes:
                rack_capacities[n.rack_id] = rack_capacities.get(n.rack_id, 0.0) + sum(n.remaining_capacity().values())
            best_rack = max(rack_capacities, key=rack_capacities.get) if rack_capacities else ""
            candidate_nodes = [n for n in nodes if n.rack_id == best_rack]

        for pod in gang_job.pods:
            best_node_id: Optional[str] = None
            best_score = -1.0

            for node in candidate_nodes:
                score = self.score_node(node, pod)
                if score > best_score:
                    best_score = score
                    best_node_id = node.node_id

            if best_node_id is not None:
                trial_assignments[pod.pod_id] = best_node_id
                node_map[best_node_id].allocate(pod.demands)
                rollback_log.append((best_node_id, pod.demands))
            else:
                gang_success = False
                break

        if not gang_success:
            # Atomic rollback: revert all tentative allocations
            for node_id, demands in rollback_log:
                node_map[node_id].deallocate(demands)
            trial_assignments.clear()
            unassigned = [p.pod_id for p in gang_job.pods]
        else:
            unassigned = []

        node_utils, stranded = self._compute_metrics(nodes)
        active_nodes = sum(1 for n in nodes if any(n.allocated.values()))
        solve_time_ms = (time.perf_counter() - start_time) * 1000.0

        return PackingResult(
            assignments=trial_assignments,
            unassigned_pods=unassigned,
            node_utilization=node_utils,
            stranded_resources=stranded,
            gang_success=gang_success,
            solve_time_ms=solve_time_ms,
            total_active_nodes=active_nodes,
        )

    def _compute_metrics(self, nodes: List[HostNode]) -> Tuple[Dict[str, float], Dict[str, float]]:
        node_utils: Dict[str, float] = {}
        total_caps: Dict[str, float] = {}
        total_used: Dict[str, float] = {}

        for n in nodes:
            u_vals = []
            for k, cap in n.capacities.items():
                used = n.allocated.get(k, 0.0)
                total_caps[k] = total_caps.get(k, 0.0) + cap
                total_used[k] = total_used.get(k, 0.0) + used
                u_vals.append(used / cap if cap > 0 else 0.0)
            node_utils[n.node_id] = sum(u_vals) / len(u_vals) if u_vals else 0.0

        # Stranded capacity: if dominant resource (e.g. GPU) is fully allocated
        # but secondary resources (RAM, CPU) remain idle on that host.
        stranded: Dict[str, float] = {}
        dominant_util = max((total_used[k] / total_caps[k] for k in total_caps if total_caps[k] > 0), default=0.0)

        for k, cap in total_caps.items():
            if cap <= 0:
                continue
            util = total_used[k] / cap
            # Stranded percentage is unutilized capacity locked behind dominant bottleneck
            stranded[k] = max(0.0, dominant_util - util)

        return node_utils, stranded
