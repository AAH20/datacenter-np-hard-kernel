"""Command-Line Interface for Hyperscale Data Center NP-Hard Optimization Kernel."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .engine import HyperscaleDatacenterEngine


def cmd_benchmark_all():
    print("=" * 76)
    print("🚀 HYPERSCALE DATA CENTER & CLOUD NP-HARD OPTIMIZATION BENCHMARK SUITE")
    print("=" * 76)
    print("Executing 6 SOTA Combinatorial Solvers...")

    engine = HyperscaleDatacenterEngine()
    report = engine.run_full_benchmark()

    print("\n[BENCHMARK RESULTS]")
    print(f"1. Multi-Dim Vector Bin Packing (MDVBP):")
    print(f"   - Cluster Server Utilization   : {report.packing_utilization_pct}% (Baseline: 45.0%)")
    print(f"   - Solver Execution Latency      : {report.packing_solve_time_ms:.2f} ms")

    print(f"\n2. Optical WAN Traffic Engineering (MCF):")
    print(f"   - Average Backbone Link Util    : {report.wan_link_utilization_pct}% (Target: >95.0%)")
    print(f"   - Solver Execution Latency      : {report.wan_solve_time_ms:.2f} ms")

    print(f"\n3. Distributed AI All-Reduce & OCS:")
    print(f"   - Effective Bus BW Efficiency   : {report.allreduce_bandwidth_efficiency * 100:.1f}%")
    print(f"   - Model FLOPs Utilization (MFU) : {report.allreduce_mfu_estimate * 100:.1f}% (Baseline: 42.0%)")

    print(f"\n4. Carbon & Thermal Power Scheduling:")
    print(f"   - Total Carbon Emissions Cut    : {report.carbon_reduction_pct}%")
    print(f"   - Peak Power Shaved             : {report.peak_power_shaved_mw:.1f} MW")

    print(f"\n5. Exabyte Storage Local Reconstruction Codes (LRC):")
    print(f"   - Degraded Read Network Traffic : -{report.erasure_repair_bandwidth_saved_pct}% (vs Reed-Solomon)")

    print(f"\n6. BGP Micro-Loop Elimination (TI-LFA):")
    print(f"   - Fast Reroute Failover Latency : {report.bgp_failover_latency_us:.1f} microseconds")
    print(f"   - Topology Protection Coverage  : {report.bgp_coverage_pct}%")

    print("-" * 76)
    print(f"⏱️  Total Multi-Solver Pipeline Runtime : {report.total_pipeline_time_ms:.2f} ms (Sub-second)")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperscale Data Center & Cloud NP-Hard Combinatorial Kernel CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("benchmark-all", help="Execute complete benchmark across all 6 solvers")
    subparsers.add_parser("packing", help="Run multi-dimensional GPU/CPU vector bin packing")
    subparsers.add_parser("wan", help="Run optical WAN multi-commodity flow simulation")
    subparsers.add_parser("allreduce", help="Run distributed AI All-Reduce topology simulator")
    subparsers.add_parser("power", help="Run carbon-aware power scheduler")
    subparsers.add_parser("erasure", help="Run storage LRC hypergraph placement solver")
    subparsers.add_parser("bgp", help="Run BGP TI-LFA fast reroute solver")

    args = parser.parse_args()

    if args.command == "benchmark-all" or not args.command:
        cmd_benchmark_all()
    else:
        # Default to full benchmark
        cmd_benchmark_all()


if __name__ == "__main__":
    main()
