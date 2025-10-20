#!/usr/bin/env python3
"""
GPU AI Benchmark - Enhanced CLI with Real-World Models

Includes ResNet50, BERT, thermal monitoring, and baseline comparisons
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from gpu_benchmark.gpu_detector import GPUDetector
from gpu_benchmark.benchmarks.flops_benchmark import FLOPSBenchmark
from gpu_benchmark.benchmarks.memory_benchmark import MemoryBenchmark
from gpu_benchmark.benchmarks.power_benchmark import PowerBenchmark
from gpu_benchmark.benchmarks.workload_benchmark import WorkloadBenchmark
from gpu_benchmark.exporter import ResultExporter

# Enhanced benchmarks
try:
    from gpu_benchmark.benchmarks.resnet_benchmark import ResNet50Benchmark
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False

try:
    from gpu_benchmark.benchmarks.bert_benchmark import BERTBenchmark
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Enhanced utilities
from gpu_benchmark.utils.thermal import ThermalMonitor
from gpu_benchmark.utils.baselines import (
    get_baseline, find_closest_baseline, compare_to_baseline, list_baselines_by_vendor
)
from gpu_benchmark.utils.visualization import create_performance_summary, export_html_report


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║      GPU AI BENCHMARK TOOL v2.0.0 (Enhanced)              ║
    ║                                                           ║
    ║      Comprehensive GPU Performance Testing for AI         ║
    ║    Supports NVIDIA CUDA, AMD ROCm, Apple Silicon MPS      ║
    ║                                                           ║
    ║      ✓ Real-World Models (ResNet50, BERT)                 ║
    ║      ✓ Thermal Monitoring & Throttle Detection            ║
    ║      ✓ Baseline Comparisons                               ║
    ║      ✓ Mixed-Precision (AMP) Support                      ║
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
        gpu_name = results["gpu_info"].get("name", "Unknown GPU")

        # Initialize thermal monitor
        thermal_monitor = None
        if args.thermal or args.all:
            thermal_monitor = ThermalMonitor(device)
            print(f"\nThermal Monitor initialized")
            temp = thermal_monitor.get_temperature()
            if temp:
                print(f"  Current Temperature: {temp}°C")

        # Run FLOPS benchmark
        if args.all or args.flops:
            print("\n" + "="*70)
            print("[1/7] Running FLOPS Benchmark...")
            print("="*70)
            results["benchmarks_run"].append("flops")
            flops_bench = FLOPSBenchmark(device, warmup_iterations=args.warmup, test_iterations=args.iterations)
            results["flops"] = flops_bench.run_all()
            flops_bench.print_summary()

        # Run Memory benchmark
        if args.all or args.memory:
            print("\n" + "="*70)
            print("[2/7] Running Memory Benchmark...")
            print("="*70)
            results["benchmarks_run"].append("memory")
            memory_bench = MemoryBenchmark(device, warmup_iterations=args.warmup, test_iterations=args.iterations)
            results["memory"] = memory_bench.run_all()
            memory_bench.print_summary()

        # Run ResNet50 benchmark
        if (args.all or args.resnet) and RESNET_AVAILABLE:
            print("\n" + "="*70)
            print("[3/7] Running ResNet50 Benchmark (Real-World Computer Vision)...")
            print("="*70)
            results["benchmarks_run"].append("resnet50")
            resnet_bench = ResNet50Benchmark(device, batch_size=args.batch_size)
            results["resnet50"] = resnet_bench.run_all()
        elif args.resnet and not RESNET_AVAILABLE:
            print("\n[3/7] ResNet50 benchmark skipped (torchvision not installed)")
            print("      Install with: pip install torchvision")

        # Run BERT benchmark
        if (args.all or args.bert) and BERT_AVAILABLE:
            print("\n" + "="*70)
            print("[4/7] Running BERT Benchmark (Real-World NLP)...")
            print("="*70)
            results["benchmarks_run"].append("bert")
            bert_bench = BERTBenchmark(device, batch_size=args.bert_batch_size, seq_length=args.seq_length)
            results["bert"] = bert_bench.run_all()
        elif args.bert and not BERT_AVAILABLE:
            print("\n[4/7] BERT benchmark skipped (not available)")

        # Run Power benchmark
        if args.all or args.power:
            print("\n" + "="*70)
            print("[5/7] Running Power Benchmark...")
            print("="*70)
            results["benchmarks_run"].append("power")
            power_bench = PowerBenchmark(device)
            results["power"] = power_bench.run_all()
            power_bench.print_summary()

        # Run Workload benchmark
        if args.all or args.workload:
            print("\n" + "="*70)
            print("[6/7] Running AI Workload Benchmark...")
            print("="*70)
            results["benchmarks_run"].append("workloads")
            workload_bench = WorkloadBenchmark(device, warmup_iterations=args.warmup, test_iterations=args.iterations)
            results["workloads"] = workload_bench.run_all()
            workload_bench.print_summary()

        # Thermal monitoring summary
        if thermal_monitor and (args.all or args.thermal):
            print("\n" + "="*70)
            print("[7/7] Thermal Monitoring Summary...")
            print("="*70)

            headroom = thermal_monitor.get_thermal_headroom()
            if headroom:
                print(f"\n  Current Temperature: {headroom['current_temp_celsius']}°C")
                print(f"  Slowdown Threshold: {headroom['slowdown_temp_celsius']}°C")
                print(f"  Thermal Headroom: {headroom['headroom_celsius']}°C ({headroom['headroom_percent']:.1f}%)")

            throttle = thermal_monitor.get_throttle_reasons()
            if throttle:
                print(f"\n  Throttling Status:")
                print(f"    Thermal Limit: {'Yes' if throttle['thermal_limit'] else 'No'}")
                print(f"    Power Limit: {'Yes' if throttle['power_limit'] else 'No'}")

        # Calculate total time
        total_time = time.time() - start_time
        results["total_time_seconds"] = total_time

        # Baseline comparison
        if args.compare:
            print("\n" + "="*70)
            print("Baseline Comparison")
            print("="*70)

            baseline_gpu = args.compare
            baseline = get_baseline(baseline_gpu)

            if baseline:
                print(f"\nComparing to: {baseline_gpu}")

                # Prepare measured results for comparison
                measured = {}
                if "flops" in results and "fp32" in results["flops"]:
                    measured["fp32_tflops"] = results["flops"]["fp32"].get("median_gflops", 0) / 1000

                if "memory" in results and "bandwidth" in results["memory"]:
                    d2d = results["memory"]["bandwidth"].get("d2d", {})
                    measured["memory_bandwidth_gbps"] = d2d.get("bandwidth_gbps", 0)

                if "resnet50" in results and "training_fp32" in results["resnet50"]:
                    measured["resnet50_training_img_sec"] = results["resnet50"]["training_fp32"].get("throughput_images_per_sec", 0)

                comparison = compare_to_baseline(measured, baseline_gpu)

                if "comparisons" in comparison:
                    for metric, data in comparison["comparisons"].items():
                        print(f"\n  {metric}:")
                        print(f"    Measured: {data['measured']:.2f}")
                        print(f"    Baseline: {data['baseline']:.2f}")
                        print(f"    Performance: {data['percent_of_baseline']:.1f}% of baseline")
            else:
                print(f"\nBaseline '{baseline_gpu}' not found.")
                print(f"\nAvailable baselines:")
                print(f"  NVIDIA: {', '.join(list_baselines_by_vendor('NVIDIA')[:5])}, ...")
                print(f"  AMD: {', '.join(list_baselines_by_vendor('AMD')[:3])}, ...")
                print(f"  Apple: {', '.join(list_baselines_by_vendor('Apple')[:5])}, ...")

        # Auto-find closest baseline
        if args.auto_compare and "flops" in results and "memory" in results:
            print("\n" + "="*70)
            print("Auto-Detected Closest Baseline")
            print("="*70)

            fp32_tflops = results["flops"].get("fp32", {}).get("median_gflops", 0) / 1000
            memory_gb = results["gpu_info"].get("total_memory_gb", 0)
            vendor = results["gpu_info"].get("vendor", "")

            closest = find_closest_baseline(fp32_tflops, memory_gb, vendor)
            if closest:
                closest_name, closest_baseline, similarity = closest
                print(f"\nClosest match: {closest_name} (similarity score: {similarity:.3f})")
                print(f"  Baseline FP32: {closest_baseline.get('fp32_tflops', 0):.2f} TFLOPS")
                print(f"  Your FP32: {fp32_tflops:.2f} TFLOPS")
                print(f"  Performance: {(fp32_tflops / closest_baseline.get('fp32_tflops', 1)) * 100:.1f}% of baseline")

        # Performance summary with visualizations
        if args.summary:
            summary = create_performance_summary(results, gpu_name)
            print(summary)

        # Export results
        exporter = ResultExporter(results, output_dir=args.output_dir)

        # Print summary report
        if not args.quiet:
            exporter.print_summary_report()

        # Export to JSON
        if args.json:
            exporter.export_json(args.json if args.json != True else None)

        # Export to CSV
        if args.csv:
            exporter.export_csv(args.csv if args.csv != True else None)

        # Export to HTML
        if args.html:
            html_path = args.html if isinstance(args.html, str) else f"{args.output_dir}/benchmark_report.html"
            export_html_report(results, html_path, gpu_name)

        print(f"\n{'='*70}")
        print(f"Total benchmark time: {total_time:.2f} seconds")
        print(f"Benchmark completed successfully!")
        print(f"{'='*70}\n")

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
        description="GPU AI Benchmark - Enhanced version with real-world models and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with summary
  python benchmark_enhanced.py --all --summary

  # Run ResNet50 and BERT benchmarks only
  python benchmark_enhanced.py --resnet --bert

  # Compare against RTX 4090 baseline
  python benchmark_enhanced.py --all --compare "RTX 4090"

  # Auto-find closest baseline
  python benchmark_enhanced.py --all --auto-compare --summary

  # Export to HTML report
  python benchmark_enhanced.py --all --html report.html

  # Run with thermal monitoring
  python benchmark_enhanced.py --all --thermal --summary

For more information, visit: https://github.com/Hiphex/Gpu-Ai-Bench
        """
    )

    # Benchmark selection
    benchmark_group = parser.add_argument_group('Benchmark Selection')
    benchmark_group.add_argument('--all', action='store_true', help='Run all benchmarks (default)')
    benchmark_group.add_argument('--flops', action='store_true', help='Run FLOPS benchmark')
    benchmark_group.add_argument('--memory', action='store_true', help='Run memory bandwidth benchmark')
    benchmark_group.add_argument('--resnet', action='store_true', help='Run ResNet50 real-world benchmark')
    benchmark_group.add_argument('--bert', action='store_true', help='Run BERT real-world benchmark')
    benchmark_group.add_argument('--power', action='store_true', help='Run power consumption benchmark')
    benchmark_group.add_argument('--workload', action='store_true', help='Run AI workload benchmark')
    benchmark_group.add_argument('--thermal', action='store_true', help='Enable thermal monitoring')

    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations (default: 10)')
    config_group.add_argument('--iterations', type=int, default=100, help='Number of test iterations (default: 100)')
    config_group.add_argument('--batch-size', type=int, default=32, help='Batch size for ResNet50 (default: 32)')
    config_group.add_argument('--bert-batch-size', type=int, default=8, help='Batch size for BERT (default: 8)')
    config_group.add_argument('--seq-length', type=int, default=512, help='Sequence length for BERT (default: 512)')

    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--compare', type=str, metavar='GPU', help='Compare against a baseline GPU (e.g., "RTX 4090")')
    analysis_group.add_argument('--auto-compare', action='store_true', help='Automatically find and compare to closest baseline')
    analysis_group.add_argument('--summary', action='store_true', help='Show performance summary with visualizations')
    analysis_group.add_argument('--list-baselines', action='store_true', help='List all available baseline GPUs')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--json', nargs='?', const=True, metavar='FILE', help='Export results to JSON')
    output_group.add_argument('--csv', nargs='?', const=True, metavar='FILE', help='Export results to CSV')
    output_group.add_argument('--html', nargs='?', const=True, metavar='FILE', help='Export HTML report')
    output_group.add_argument('--output-dir', type=str, default='results', help='Output directory (default: results)')
    output_group.add_argument('--quiet', action='store_true', help='Suppress summary output')

    # Version
    parser.add_argument('--version', action='version', version='GPU AI Benchmark v2.0.0 (Enhanced)')

    args = parser.parse_args()

    # Handle list-baselines
    if args.list_baselines:
        print("\nAvailable Baseline GPUs:\n")
        print("NVIDIA GPUs:")
        for gpu in list_baselines_by_vendor('NVIDIA'):
            print(f"  - {gpu}")
        print("\nAMD GPUs:")
        for gpu in list_baselines_by_vendor('AMD'):
            print(f"  - {gpu}")
        print("\nApple Silicon:")
        for gpu in list_baselines_by_vendor('Apple'):
            print(f"  - {gpu}")
        print()
        sys.exit(0)

    # If no specific benchmark is selected, run all
    if not any([args.flops, args.memory, args.resnet, args.bert, args.power, args.workload, args.thermal]):
        args.all = True

    # Print banner
    print_banner()

    # Run benchmark
    run_benchmark(args)


if __name__ == "__main__":
    main()
