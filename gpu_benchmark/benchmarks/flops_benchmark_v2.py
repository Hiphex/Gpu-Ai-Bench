"""
Enhanced FLOPS Benchmark Module
Measures TFLOPS with proper timing, precision control, and statistics
Supports FP32, TF32, BF16, FP16, and FP8 operations
"""

import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DeviceTimer, benchmark_operation, MatmulPrecisionContext, get_matmul_precision_info


class EnhancedFLOPSBenchmark:
    """Enhanced FLOPS benchmark with proper timing and precision control"""

    def __init__(self, device: torch.device, warmup_iterations: int = 10, test_iterations: int = 100):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

    def run_all(self, include_tf32: bool = True, include_bf16: bool = True) -> Dict:
        """Run all FLOPS benchmarks"""
        print("\n" + "="*60)
        print("ENHANCED FLOPS BENCHMARK")
        print("="*60)

        # Record precision settings
        self.results["precision_info"] = get_matmul_precision_info()
        print(f"\nPrecision Settings:")
        for key, value in self.results["precision_info"].items():
            print(f"  {key}: {value}")

        # Test FP32 (true FP32, not TF32)
        print(f"\n{'='*60}")
        self.results["fp32"] = self._benchmark_matmul(torch.float32, "FP32 (True)", use_tf32=False)

        # Test TF32 (on NVIDIA Ampere+)
        if include_tf32 and self.device.type == 'cuda':
            self.results["tf32"] = self._benchmark_matmul(torch.float32, "TF32", use_tf32=True)

        # Test BF16
        if include_bf16:
            try:
                self.results["bf16"] = self._benchmark_matmul(torch.bfloat16, "BF16")
            except Exception as e:
                print(f"BF16 not supported: {e}")

        # Test FP16
        self.results["fp16"] = self._benchmark_matmul(torch.float16, "FP16")

        # Test FP8 if available
        try:
            if hasattr(torch, 'float8_e4m3fn'):
                self.results["fp8"] = self._benchmark_matmul(torch.float8_e4m3fn, "FP8")
        except Exception as e:
            print(f"FP8 not available: {e}")

        return self.results

    def _benchmark_matmul(self, dtype, dtype_name: str, use_tf32: bool = False) -> Dict:
        """Benchmark matrix multiplication for a specific data type"""
        print(f"\n{'='*60}")
        print(f"Benchmarking {dtype_name} Matrix Multiplication")
        print(f"{'='*60}")

        matrix_sizes = [
            (4096, 4096),
            (8192, 8192),
        ]

        results = []

        for M, N in matrix_sizes:
            K = N
            print(f"\nMatrix size: {M}x{K} × {K}x{N}")

            try:
                # Create matrices
                if dtype == torch.int8:
                    A = torch.randint(-128, 127, (M, K), dtype=dtype, device=self.device)
                    B = torch.randint(-128, 127, (K, N), dtype=dtype, device=self.device)
                else:
                    A = torch.randn(M, K, dtype=dtype, device=self.device)
                    B = torch.randn(K, N, dtype=dtype, device=self.device)

                # Set up precision context for TF32
                precision_ctx = MatmulPrecisionContext("high" if use_tf32 else "highest", self.device)

                # Define operation
                def matmul_op():
                    if dtype == torch.int8:
                        return torch.matmul(A.to(torch.float32), B.to(torch.float32))
                    else:
                        return torch.matmul(A, B)

                # Benchmark with proper timing and statistics
                with precision_ctx:
                    stats = benchmark_operation(
                        matmul_op,
                        warmup_iterations=self.warmup_iterations,
                        test_iterations=self.test_iterations,
                        device=self.device,
                        return_stats=True
                    )

                # Calculate TFLOPS using median time
                flops = 2 * M * N * K
                median_time_s = stats['median_ms'] / 1000.0
                mean_time_s = stats['mean_ms'] / 1000.0

                median_tflops = (flops / median_time_s) / 1e12
                mean_tflops = (flops / mean_time_s) / 1e12

                result = {
                    "matrix_size": f"{M}x{K}x{N}",
                    "median_tflops": median_tflops,
                    "mean_tflops": mean_tflops,
                    "median_ms": stats['median_ms'],
                    "mean_ms": stats['mean_ms'],
                    "std_ms": stats['std_ms'],
                    "p10_ms": stats['p10_ms'],
                    "p90_ms": stats['p90_ms'],
                    "min_ms": stats['min_ms'],
                    "max_ms": stats['max_ms'],
                }

                results.append(result)

                print(f"  Median: {median_tflops:.2f} TFLOPS ({stats['median_ms']:.2f} ms)")
                print(f"  Mean:   {mean_tflops:.2f} TFLOPS ({stats['mean_ms']:.2f} ms)")
                print(f"  Std:    {stats['std_ms']:.2f} ms")
                print(f"  Range:  [{stats['p10_ms']:.2f} - {stats['p90_ms']:.2f}] ms (p10-p90)")

                # Clean up
                del A, B
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()

            except RuntimeError as e:
                print(f"  Skipped (error: {e})")
                results.append({
                    "matrix_size": f"{M}x{K}x{N}",
                    "error": str(e)
                })

        # Calculate aggregate statistics
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            median_tflops_values = [r["median_tflops"] for r in valid_results]
            mean_tflops_values = [r["mean_tflops"] for r in valid_results]

            aggregate = {
                "dtype": dtype_name,
                "use_tf32": use_tf32,
                "median_tflops": np.median(median_tflops_values),
                "mean_tflops": np.mean(mean_tflops_values),
                "max_tflops": np.max(median_tflops_values),
                "detailed_results": results
            }
        else:
            aggregate = {
                "dtype": dtype_name,
                "use_tf32": use_tf32,
                "median_tflops": 0,
                "mean_tflops": 0,
                "max_tflops": 0,
                "detailed_results": results
            }

        return aggregate

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print a summary of FLOPS results"""
        print("\n" + "="*60)
        print("ENHANCED FLOPS BENCHMARK SUMMARY")
        print("="*60)
        for precision, data in self.results.items():
            if precision == "precision_info":
                continue
            median = data.get('median_tflops', 0)
            mean = data.get('mean_tflops', 0)
            max_perf = data.get('max_tflops', 0)
            print(f"{precision.upper():.<20} Median: {median:>7.2f} | Mean: {mean:>7.2f} | Peak: {max_perf:>7.2f} TFLOPS")
        print("="*60)
