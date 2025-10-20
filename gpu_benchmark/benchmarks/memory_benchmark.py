"""
Memory Benchmark Module
Measures memory bandwidth and capacity
"""

import torch
import time
from typing import Dict


class MemoryBenchmark:
    """Benchmark GPU memory performance"""

    def __init__(self, device: torch.device, warmup_iterations: int = 5, test_iterations: int = 50):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

    def run_all(self) -> Dict:
        """Run all memory benchmarks"""
        print("\n" + "="*60)
        print("MEMORY BENCHMARK")
        print("="*60)

        # Get memory info
        self.results["memory_info"] = self._get_memory_info()

        # Test memory bandwidth
        self.results["bandwidth"] = self._benchmark_bandwidth()

        return self.results

    def _get_memory_info(self) -> Dict:
        """Get GPU memory information"""
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device)
        reserved_memory = torch.cuda.memory_reserved(self.device)

        info = {
            "total_gb": total_memory / (1024**3),
            "allocated_gb": allocated_memory / (1024**3),
            "reserved_gb": reserved_memory / (1024**3),
            "free_gb": (total_memory - allocated_memory) / (1024**3)
        }

        print(f"\nMemory Information:")
        print(f"  Total Memory: {info['total_gb']:.2f} GB")
        print(f"  Allocated: {info['allocated_gb']:.2f} GB")
        print(f"  Free: {info['free_gb']:.2f} GB")

        return info

    def _benchmark_bandwidth(self) -> Dict:
        """Benchmark memory bandwidth"""
        print(f"\nBenchmarking Memory Bandwidth...")

        results = {}

        # Test copy operations (host to device, device to host, device to device)
        data_sizes_mb = [128, 512, 1024, 2048]  # Different data sizes in MB

        for size_mb in data_sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            num_elements = size_bytes // 4  # float32 = 4 bytes

            print(f"\n  Data size: {size_mb} MB")

            # Device to Device (D2D) bandwidth
            try:
                d2d_bandwidth = self._measure_d2d_bandwidth(num_elements)
                print(f"    Device-to-Device: {d2d_bandwidth:.2f} GB/s")
            except Exception as e:
                d2d_bandwidth = 0
                print(f"    Device-to-Device: Error - {e}")

            # Host to Device (H2D) bandwidth
            try:
                h2d_bandwidth = self._measure_h2d_bandwidth(num_elements)
                print(f"    Host-to-Device: {h2d_bandwidth:.2f} GB/s")
            except Exception as e:
                h2d_bandwidth = 0
                print(f"    Host-to-Device: Error - {e}")

            # Device to Host (D2H) bandwidth
            try:
                d2h_bandwidth = self._measure_d2h_bandwidth(num_elements)
                print(f"    Device-to-Host: {d2h_bandwidth:.2f} GB/s")
            except Exception as e:
                d2h_bandwidth = 0
                print(f"    Device-to-Host: Error - {e}")

            results[f"{size_mb}mb"] = {
                "device_to_device_gbs": d2d_bandwidth,
                "host_to_device_gbs": h2d_bandwidth,
                "device_to_host_gbs": d2h_bandwidth
            }

        # Calculate peak bandwidth (typically the largest D2D measurement)
        peak_bandwidth = max(
            results[size]["device_to_device_gbs"]
            for size in results
            if results[size]["device_to_device_gbs"] > 0
        ) if results else 0

        return {
            "peak_bandwidth_gbs": peak_bandwidth,
            "detailed_results": results
        }

    def _measure_d2d_bandwidth(self, num_elements: int) -> float:
        """Measure device-to-device memory bandwidth"""
        src = torch.randn(num_elements, dtype=torch.float32, device=self.device)
        dst = torch.empty_like(src)

        # Warmup
        for _ in range(self.warmup_iterations):
            dst.copy_(src)
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.test_iterations):
            dst.copy_(src)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_time = (end_time - start_time) / self.test_iterations
        bytes_transferred = num_elements * 4  # float32 = 4 bytes
        bandwidth_gbs = (bytes_transferred / elapsed_time) / (1024**3)

        del src, dst
        torch.cuda.empty_cache()

        return bandwidth_gbs

    def _measure_h2d_bandwidth(self, num_elements: int) -> float:
        """Measure host-to-device memory bandwidth"""
        src = torch.randn(num_elements, dtype=torch.float32, device='cpu', pin_memory=True)

        # Warmup
        for _ in range(self.warmup_iterations):
            dst = src.to(self.device, non_blocking=False)
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.test_iterations):
            dst = src.to(self.device, non_blocking=False)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_time = (end_time - start_time) / self.test_iterations
        bytes_transferred = num_elements * 4
        bandwidth_gbs = (bytes_transferred / elapsed_time) / (1024**3)

        del src, dst
        torch.cuda.empty_cache()

        return bandwidth_gbs

    def _measure_d2h_bandwidth(self, num_elements: int) -> float:
        """Measure device-to-host memory bandwidth"""
        src = torch.randn(num_elements, dtype=torch.float32, device=self.device)

        # Warmup
        for _ in range(self.warmup_iterations):
            dst = src.to('cpu', non_blocking=False)
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.test_iterations):
            dst = src.to('cpu', non_blocking=False)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_time = (end_time - start_time) / self.test_iterations
        bytes_transferred = num_elements * 4
        bandwidth_gbs = (bytes_transferred / elapsed_time) / (1024**3)

        del src, dst
        torch.cuda.empty_cache()

        return bandwidth_gbs

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print a summary of memory results"""
        print("\n" + "="*60)
        print("MEMORY BENCHMARK SUMMARY")
        print("="*60)
        mem_info = self.results.get("memory_info", {})
        bandwidth = self.results.get("bandwidth", {})

        print(f"Total Memory:............... {mem_info.get('total_gb', 0):.2f} GB")
        print(f"Peak Bandwidth:............. {bandwidth.get('peak_bandwidth_gbs', 0):.2f} GB/s")
        print("="*60)
