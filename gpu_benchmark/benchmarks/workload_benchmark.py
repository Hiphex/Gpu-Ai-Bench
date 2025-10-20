"""
Real-world AI Workload Benchmark Module
Tests GPU performance with common AI operations
"""

import torch
import torch.nn as nn
import time
from typing import Dict


class WorkloadBenchmark:
    """Benchmark real-world AI workloads"""

    def __init__(self, device: torch.device, warmup_iterations: int = 5, test_iterations: int = 50):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

    def run_all(self) -> Dict:
        """Run all workload benchmarks"""
        print("\n" + "="*60)
        print("AI WORKLOAD BENCHMARK")
        print("="*60)

        # Test convolution operations (CNN workloads)
        self.results["convolution"] = self._benchmark_convolution()

        # Test transformer operations (attention mechanism)
        self.results["attention"] = self._benchmark_attention()

        # Test element-wise operations
        self.results["elementwise"] = self._benchmark_elementwise()

        # Test reduction operations
        self.results["reduction"] = self._benchmark_reduction()

        return self.results

    def _benchmark_convolution(self) -> Dict:
        """Benchmark 2D convolution operations (common in CNNs)"""
        print("\nBenchmarking Convolution Operations...")

        configs = [
            # (batch_size, channels_in, channels_out, image_size, kernel_size)
            (32, 3, 64, 224, 3),      # Initial layer of ResNet-like model
            (32, 64, 128, 112, 3),    # Mid layer
            (32, 256, 512, 28, 3),    # Deep layer
        ]

        results = []

        for batch_size, in_channels, out_channels, img_size, kernel_size in configs:
            print(f"  Config: batch={batch_size}, in_ch={in_channels}, out_ch={out_channels}, size={img_size}x{img_size}")

            try:
                # Create conv layer
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1).to(self.device)
                input_tensor = torch.randn(batch_size, in_channels, img_size, img_size, device=self.device)

                # Warmup
                for _ in range(self.warmup_iterations):
                    output = conv(input_tensor)
                    torch.cuda.synchronize()

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(self.test_iterations):
                    output = conv(input_tensor)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) / self.test_iterations
                throughput = batch_size / elapsed_time  # images per second

                results.append({
                    "config": f"B{batch_size}_IC{in_channels}_OC{out_channels}_S{img_size}",
                    "time_ms": elapsed_time * 1000,
                    "throughput_img_per_sec": throughput
                })

                print(f"    Time: {elapsed_time * 1000:.2f} ms")
                print(f"    Throughput: {throughput:.2f} img/s")

                del conv, input_tensor, output
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error: {e}")
                results.append({"config": f"B{batch_size}_IC{in_channels}_OC{out_channels}_S{img_size}", "error": str(e)})

        avg_throughput = sum(r["throughput_img_per_sec"] for r in results if "error" not in r) / len([r for r in results if "error" not in r]) if results else 0

        return {
            "average_throughput_img_per_sec": avg_throughput,
            "detailed_results": results
        }

    def _benchmark_attention(self) -> Dict:
        """Benchmark attention mechanism (common in Transformers)"""
        print("\nBenchmarking Attention Operations...")

        configs = [
            # (batch_size, seq_length, embed_dim, num_heads)
            (32, 128, 512, 8),   # Small transformer
            (16, 512, 768, 12),  # Medium (BERT-like)
            (8, 1024, 1024, 16), # Large
        ]

        results = []

        for batch_size, seq_len, embed_dim, num_heads in configs:
            print(f"  Config: batch={batch_size}, seq_len={seq_len}, embed={embed_dim}, heads={num_heads}")

            try:
                # Create multi-head attention layer
                attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(self.device)
                query = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                key = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                value = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

                # Warmup
                for _ in range(self.warmup_iterations):
                    output, _ = attention(query, key, value)
                    torch.cuda.synchronize()

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(self.test_iterations):
                    output, _ = attention(query, key, value)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) / self.test_iterations
                throughput = (batch_size * seq_len) / elapsed_time  # tokens per second

                results.append({
                    "config": f"B{batch_size}_S{seq_len}_E{embed_dim}_H{num_heads}",
                    "time_ms": elapsed_time * 1000,
                    "throughput_tokens_per_sec": throughput
                })

                print(f"    Time: {elapsed_time * 1000:.2f} ms")
                print(f"    Throughput: {throughput:.2f} tokens/s")

                del attention, query, key, value, output
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error: {e}")
                results.append({"config": f"B{batch_size}_S{seq_len}_E{embed_dim}_H{num_heads}", "error": str(e)})

        avg_throughput = sum(r["throughput_tokens_per_sec"] for r in results if "error" not in r) / len([r for r in results if "error" not in r]) if results else 0

        return {
            "average_throughput_tokens_per_sec": avg_throughput,
            "detailed_results": results
        }

    def _benchmark_elementwise(self) -> Dict:
        """Benchmark element-wise operations (activations, etc.)"""
        print("\nBenchmarking Element-wise Operations...")

        size = 100 * 1024 * 1024  # 100M elements
        print(f"  Tensor size: {size:,} elements")

        operations = {
            "ReLU": lambda x: torch.relu(x),
            "GELU": lambda x: torch.nn.functional.gelu(x),
            "Sigmoid": lambda x: torch.sigmoid(x),
            "Tanh": lambda x: torch.tanh(x),
        }

        results = {}

        for op_name, op_func in operations.items():
            try:
                input_tensor = torch.randn(size, device=self.device)

                # Warmup
                for _ in range(self.warmup_iterations):
                    output = op_func(input_tensor)
                    torch.cuda.synchronize()

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(self.test_iterations):
                    output = op_func(input_tensor)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) / self.test_iterations
                throughput_gb_s = (size * 4 * 2) / elapsed_time / (1024**3)  # Read + write, 4 bytes per float

                results[op_name.lower()] = {
                    "time_ms": elapsed_time * 1000,
                    "throughput_gb_per_sec": throughput_gb_s
                }

                print(f"  {op_name}: {elapsed_time * 1000:.2f} ms, {throughput_gb_s:.2f} GB/s")

                del input_tensor, output
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  {op_name}: Error - {e}")
                results[op_name.lower()] = {"error": str(e)}

        return results

    def _benchmark_reduction(self) -> Dict:
        """Benchmark reduction operations"""
        print("\nBenchmarking Reduction Operations...")

        size = 100 * 1024 * 1024  # 100M elements
        print(f"  Tensor size: {size:,} elements")

        operations = {
            "Sum": lambda x: torch.sum(x),
            "Mean": lambda x: torch.mean(x),
            "Max": lambda x: torch.max(x),
            "ArgMax": lambda x: torch.argmax(x),
        }

        results = {}

        for op_name, op_func in operations.items():
            try:
                input_tensor = torch.randn(size, device=self.device)

                # Warmup
                for _ in range(self.warmup_iterations):
                    output = op_func(input_tensor)
                    torch.cuda.synchronize()

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(self.test_iterations):
                    output = op_func(input_tensor)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) / self.test_iterations
                throughput_gb_s = (size * 4) / elapsed_time / (1024**3)

                results[op_name.lower()] = {
                    "time_ms": elapsed_time * 1000,
                    "throughput_gb_per_sec": throughput_gb_s
                }

                print(f"  {op_name}: {elapsed_time * 1000:.2f} ms, {throughput_gb_s:.2f} GB/s")

                del input_tensor, output
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  {op_name}: Error - {e}")
                results[op_name.lower()] = {"error": str(e)}

        return results

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print a summary of workload results"""
        print("\n" + "="*60)
        print("AI WORKLOAD BENCHMARK SUMMARY")
        print("="*60)

        conv = self.results.get("convolution", {})
        attn = self.results.get("attention", {})

        print(f"Convolution Throughput:..... {conv.get('average_throughput_img_per_sec', 0):.2f} img/s")
        print(f"Attention Throughput:....... {attn.get('average_throughput_tokens_per_sec', 0):.2f} tokens/s")
        print("="*60)
