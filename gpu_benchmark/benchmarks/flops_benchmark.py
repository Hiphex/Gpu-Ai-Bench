"""
FLOPS Benchmark Module
Measures theoretical and practical TFLOPS for FP32, FP16, and FP8 operations
"""

import torch
import time
from typing import Dict


class FLOPSBenchmark:
    """Benchmark GPU FLOPS performance"""

    def __init__(self, device: torch.device, warmup_iterations: int = 10, test_iterations: int = 100):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

    def _synchronize(self):
        """Synchronize device (works for CUDA, ROCm, and MPS)"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()

    def run_all(self) -> Dict:
        """Run all FLOPS benchmarks"""
        print("\n" + "="*60)
        print("FLOPS BENCHMARK")
        print("="*60)

        # Test different precisions
        self.results["fp32"] = self._benchmark_matmul(torch.float32, "FP32")
        self.results["fp16"] = self._benchmark_matmul(torch.float16, "FP16")

        # FP8 is not natively supported in PyTorch yet, but we can simulate with int8
        # or use torch.float8_e4m3fn if available (PyTorch 2.1+)
        try:
            if hasattr(torch, 'float8_e4m3fn'):
                self.results["fp8"] = self._benchmark_matmul(torch.float8_e4m3fn, "FP8")
            else:
                print("FP8 not available in this PyTorch version, using INT8 as approximation")
                self.results["int8"] = self._benchmark_matmul(torch.int8, "INT8")
        except Exception as e:
            print(f"Warning: Could not benchmark FP8/INT8: {e}")

        return self.results

    def _benchmark_matmul(self, dtype, dtype_name: str) -> Dict:
        """Benchmark matrix multiplication for a specific data type"""
        print(f"\nBenchmarking {dtype_name} Matrix Multiplication...")

        # Use large matrices for better GPU utilization
        matrix_sizes = [
            (4096, 4096),
            (8192, 8192),
        ]

        results = []

        for M, N in matrix_sizes:
            K = N
            print(f"  Matrix size: {M}x{K} × {K}x{N}")

            try:
                # Skip int8 for now as it requires special handling
                if dtype == torch.int8:
                    # For INT8, we use a different approach
                    A = torch.randint(-128, 127, (M, K), dtype=dtype, device=self.device)
                    B = torch.randint(-128, 127, (K, N), dtype=dtype, device=self.device)
                else:
                    A = torch.randn(M, K, dtype=dtype, device=self.device)
                    B = torch.randn(K, N, dtype=dtype, device=self.device)

                # Warmup
                for _ in range(self.warmup_iterations):
                    if dtype == torch.int8:
                        C = torch.matmul(A.to(torch.float32), B.to(torch.float32))
                    else:
                        C = torch.matmul(A, B)
                    self._synchronize()

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(self.test_iterations):
                    if dtype == torch.int8:
                        C = torch.matmul(A.to(torch.float32), B.to(torch.float32))
                    else:
                        C = torch.matmul(A, B)
                self._synchronize()
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) / self.test_iterations

                # Calculate TFLOPS
                # For matrix multiplication: FLOPs = 2 * M * N * K
                flops = 2 * M * N * K
                tflops = (flops / elapsed_time) / 1e12

                results.append({
                    "matrix_size": f"{M}x{K}x{N}",
                    "tflops": tflops,
                    "time_ms": elapsed_time * 1000
                })

                print(f"    TFLOPS: {tflops:.2f}")
                print(f"    Time: {elapsed_time * 1000:.2f} ms")

                # Clean up
                del A, B, C
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()

            except RuntimeError as e:
                print(f"    Skipped (error: {e})")
                results.append({
                    "matrix_size": f"{M}x{K}x{N}",
                    "tflops": 0,
                    "time_ms": 0,
                    "error": str(e)
                })

        # Calculate average TFLOPS
        avg_tflops = sum(r["tflops"] for r in results if "error" not in r) / len([r for r in results if "error" not in r]) if results else 0

        return {
            "dtype": dtype_name,
            "average_tflops": avg_tflops,
            "detailed_results": results
        }

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print a summary of FLOPS results"""
        print("\n" + "="*60)
        print("FLOPS BENCHMARK SUMMARY")
        print("="*60)
        for precision, data in self.results.items():
            print(f"{precision.upper():.<30} {data['average_tflops']:.2f} TFLOPS")
        print("="*60)
