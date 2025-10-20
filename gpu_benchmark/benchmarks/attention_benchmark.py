"""
Scaled Dot Product Attention Benchmark
Tests different SDPA backends (FlashAttention, Memory-Efficient, Math)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DeviceTimer, benchmark_operation


class SDPABenchmark:
    """Benchmark Scaled Dot Product Attention with different backends"""

    def __init__(self, device: torch.device, warmup_iterations: int = 5, test_iterations: int = 50):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

        # Check if SDPA is available (PyTorch 2.0+)
        self.sdpa_available = hasattr(F, 'scaled_dot_product_attention')

    def run_all(self) -> Dict:
        """Run SDPA benchmarks with different backends"""
        print("\n" + "="*60)
        print("SCALED DOT PRODUCT ATTENTION BENCHMARK")
        print("="*60)

        if not self.sdpa_available:
            print("SDPA not available (requires PyTorch 2.0+)")
            return {"available": False}

        # Test configurations (batch, seq_len, embed_dim, num_heads)
        configs = [
            (16, 512, 768, 12, "BERT-like"),
            (8, 2048, 1024, 16, "Large Context"),
        ]

        for batch, seq_len, embed_dim, num_heads, name in configs:
            print(f"\n{'-'*60}")
            print(f"Config: {name}")
            print(f"  Batch: {batch}, Seq Length: {seq_len}, Embed: {embed_dim}, Heads: {num_heads}")
            print(f"{'-'*60}")

            self.results[name] = self._benchmark_sdpa(batch, seq_len, embed_dim, num_heads)

        return self.results

    def _benchmark_sdpa(
        self,
        batch_size: int,
        seq_length: int,
        embed_dim: int,
        num_heads: int
    ) -> Dict:
        """Benchmark SDPA with different backends"""

        head_dim = embed_dim // num_heads

        # Create dummy Q, K, V tensors
        # Shape: (batch, num_heads, seq_len, head_dim)
        Q = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float16)
        K = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float16)
        V = torch.randn(batch_size, num_heads, seq_length, head_dim, device=self.device, dtype=torch.float16)

        results = {}

        # Try different backends
        backends = []

        # Check what's available
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend

            # List of backends to try
            potential_backends = [
                (SDPBackend.FLASH_ATTENTION, "FlashAttention"),
                (SDPBackend.EFFICIENT_ATTENTION, "Memory-Efficient"),
                (SDPBackend.MATH, "Math (Baseline)"),
            ]

            for backend_enum, backend_name in potential_backends:
                try:
                    # Test if backend works
                    with sdpa_kernel([backend_enum]):
                        _ = F.scaled_dot_product_attention(Q[:1], K[:1], V[:1])

                    backends.append((backend_enum, backend_name))
                except Exception as e:
                    print(f"  {backend_name}: Not available ({str(e)[:50]})")

        except ImportError:
            # Fallback: no backend selection available
            backends = [(None, "Default")]

        # Benchmark each available backend
        for backend_enum, backend_name in backends:
            try:
                def sdpa_op():
                    if backend_enum is not None:
                        from torch.nn.attention import sdpa_kernel
                        with sdpa_kernel([backend_enum]):
                            return F.scaled_dot_product_attention(Q, K, V)
                    else:
                        return F.scaled_dot_product_attention(Q, K, V)

                stats = benchmark_operation(
                    sdpa_op,
                    warmup_iterations=self.warmup_iterations,
                    test_iterations=self.test_iterations,
                    device=self.device,
                    return_stats=True
                )

                # Calculate throughput
                tokens_per_batch = batch_size * seq_length
                median_throughput = tokens_per_batch / (stats['median_ms'] / 1000.0)

                results[backend_name] = {
                    "median_ms": stats['median_ms'],
                    "mean_ms": stats['mean_ms'],
                    "throughput_tokens_per_sec": median_throughput,
                    "available": True
                }

                print(f"  {backend_name:.<30} {stats['median_ms']:>8.2f} ms ({median_throughput:>10,.0f} tokens/s)")

            except Exception as e:
                print(f"  {backend_name:.<30} Failed: {e}")
                results[backend_name] = {"available": False, "error": str(e)}

        # Clean up
        del Q, K, V
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()

        return results

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print summary of SDPA results"""
        print("\n" + "="*60)
        print("SDPA BENCHMARK SUMMARY")
        print("="*60)

        if not self.sdpa_available:
            print("SDPA not available")
            return

        for config_name, backends in self.results.items():
            print(f"\n{config_name}:")
            for backend_name, data in backends.items():
                if data.get("available", False):
                    throughput = data.get("throughput_tokens_per_sec", 0)
                    print(f"  {backend_name:.<30} {throughput:>10,.0f} tokens/s")

        print("="*60)
