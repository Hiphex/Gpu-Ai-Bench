#!/usr/bin/env python3
"""
GPU AI Benchmark - Main CLI Interface
Comprehensive GPU benchmarking tool for AI workloads
"""

import argparse
import sys
import time
from datetime import datetime

from gpu_benchmark.gpu_detector import GPUDetector
from gpu_benchmark.benchmarks.flops_benchmark import FLOPSBenchmark
from gpu_benchmark.benchmarks.memory_benchmark import MemoryBenchmark
from gpu_benchmark.benchmarks.power_benchmark import PowerBenchmark
from gpu_benchmark.benchmarks.workload_benchmark import WorkloadBenchmark
from gpu_benchmark.exporter import ResultExporter


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           GPU AI BENCHMARK TOOL v1.0.0                    ║
    ║                                                           ║
    ║      Comprehensive GPU Performance Testing for AI         ║
    ║    Supports NVIDIA CUDA, AMD ROCm, Apple Silicon MPS      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_benchmark(args):
    """Run the GPU benchmark"""
    start_time = time.time()

    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks_run": []
    }

    try:
        # Detect GPU
        print("Detecting GPU...")
        detector = GPUDetector()
        detector.print_info()
        results["gpu_info"] = detector.get_info()

        device = detector.get_device()

        # Run FLOPS benchmark
        if args.all or args.flops:
            print("\n[1/4] Running FLOPS Benchmark...")
            results["benchmarks_run"].append("flops")
            flops_bench = FLOPSBenchmark(device, warmup_iterations=args.warmup, test_iterations=args.iterations)
            results["flops"] = flops_bench.run_all()
            flops_bench.print_summary()

        # Run Memory benchmark
        if args.all or args.memory:
            print("\n[2/4] Running Memory Benchmark...")
            results["benchmarks_run"].append("memory")
            memory_bench = MemoryBenchmark(device, warmup_iterations=args.warmup, test_iterations=args.iterations)
            results["memory"] = memory_bench.run_all()
            memory_bench.print_summary()

        # Run Power benchmark
        if args.all or args.power:
            print("\n[3/4] Running Power Benchmark...")
            results["benchmarks_run"].append("power")
            power_bench = PowerBenchmark(device)
            results["power"] = power_bench.run_all()
            power_bench.print_summary()

        # Run Workload benchmark
        if args.all or args.workload:
            print("\n[4/4] Running AI Workload Benchmark...")
            results["benchmarks_run"].append("workloads")
            workload_bench = WorkloadBenchmark(device, warmup_iterations=args.warmup, test_iterations=args.iterations)
            results["workloads"] = workload_bench.run_all()
            workload_bench.print_summary()

        # Calculate total time
        total_time = time.time() - start_time
        results["total_time_seconds"] = total_time

        # Export results
        exporter = ResultExporter(results, output_dir=args.output_dir)

        # Print summary report
        exporter.print_summary_report()

        # Export to JSON
        if args.json:
            exporter.export_json(args.json if args.json != True else None)

        # Export to CSV
        if args.csv:
            exporter.export_csv(args.csv if args.csv != True else None)

        print(f"\nTotal benchmark time: {total_time:.2f} seconds")
        print("\nBenchmark completed successfully!")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GPU AI Benchmark - Comprehensive GPU performance testing for AI workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python -m gpu_benchmark.main --all

  # Run specific benchmarks
  python -m gpu_benchmark.main --flops --memory

  # Run with custom iterations and export to JSON
  python -m gpu_benchmark.main --all --iterations 200 --json

  # Export to both JSON and CSV
  python -m gpu_benchmark.main --all --json results.json --csv results.csv

For more information, visit: https://github.com/Hiphex/Gpu-Ai-Bench
        """
    )

    # Benchmark selection
    benchmark_group = parser.add_argument_group('Benchmark Selection')
    benchmark_group.add_argument('--all', action='store_true', help='Run all benchmarks (default)')
    benchmark_group.add_argument('--flops', action='store_true', help='Run FLOPS benchmark')
    benchmark_group.add_argument('--memory', action='store_true', help='Run memory bandwidth benchmark')
    benchmark_group.add_argument('--power', action='store_true', help='Run power consumption benchmark')
    benchmark_group.add_argument('--workload', action='store_true', help='Run AI workload benchmark')

    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations (default: 10)')
    config_group.add_argument('--iterations', type=int, default=100, help='Number of test iterations (default: 100)')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--json', nargs='?', const=True, metavar='FILE', help='Export results to JSON (optionally specify filename)')
    output_group.add_argument('--csv', nargs='?', const=True, metavar='FILE', help='Export results to CSV (optionally specify filename)')
    output_group.add_argument('--output-dir', type=str, default='results', help='Output directory for results (default: results)')

    # Version
    parser.add_argument('--version', action='version', version='GPU AI Benchmark v1.0.0')

    args = parser.parse_args()

    # If no specific benchmark is selected, run all
    if not any([args.flops, args.memory, args.power, args.workload]):
        args.all = True

    # Print banner
    print_banner()

    # Run benchmark
    run_benchmark(args)


if __name__ == "__main__":
    main()
