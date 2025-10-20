"""
Enhanced memory transfer benchmarking with pinned vs pageable comparison
"""

import torch
import time
from typing import Dict, Tuple


def benchmark_memory_transfer(
    size_bytes: int,
    device: torch.device,
    use_pinned: bool = True,
    iterations: int = 50
) -> Tuple[float, float]:
    """
    Benchmark H2D and D2H transfers with optional pinned memory

    Args:
        size_bytes: Size of data to transfer
        device: Target device
        use_pinned: Whether to use pinned memory (CUDA only)
        iterations: Number of iterations to average

    Returns:
        Tuple of (h2d_bandwidth_gbs, d2h_bandwidth_gbs)
    """
    num_elements = size_bytes // 4  # float32

    # Only use pinned memory for CUDA
    can_pin = (device.type == 'cuda' and use_pinned)

    # Host-to-Device
    src_cpu = torch.randn(num_elements, dtype=torch.float32, device='cpu', pin_memory=can_pin)

    # Warmup
    for _ in range(5):
        dst_gpu = src_cpu.to(device, non_blocking=can_pin)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    # Benchmark H2D
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            dst_gpu = src_cpu.to(device, non_blocking=can_pin)
        end.record()
        torch.cuda.synchronize()
        h2d_time_s = start.elapsed_time(end) / 1000.0 / iterations
    else:
        t0 = time.perf_counter()
        for _ in range(iterations):
            dst_gpu = src_cpu.to(device)
        if device.type == 'mps':
            torch.mps.synchronize()
        h2d_time_s = (time.perf_counter() - t0) / iterations

    h2d_bandwidth_gbs = (size_bytes / h2d_time_s) / (1024**3)

    # Device-to-Host
    src_gpu = torch.randn(num_elements, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(5):
        dst_cpu = src_gpu.to('cpu', non_blocking=can_pin)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    # Benchmark D2H
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            dst_cpu = src_gpu.to('cpu', non_blocking=can_pin)
        end.record()
        torch.cuda.synchronize()
        d2h_time_s = start.elapsed_time(end) / 1000.0 / iterations
    else:
        t0 = time.perf_counter()
        for _ in range(iterations):
            dst_cpu = src_gpu.to('cpu')
        if device.type == 'mps':
            torch.mps.synchronize()
        d2h_time_s = (time.perf_counter() - t0) / iterations

    d2h_bandwidth_gbs = (size_bytes / d2h_time_s) / (1024**3)

    # Cleanup
    del src_cpu, src_gpu, dst_gpu, dst_cpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

    return h2d_bandwidth_gbs, d2h_bandwidth_gbs


def compare_pinned_vs_pageable(
    size_mb: int,
    device: torch.device,
    iterations: int = 50
) -> Dict:
    """
    Compare pinned vs pageable memory transfer performance

    Args:
        size_mb: Size in megabytes
        device: Target device
        iterations: Number of iterations

    Returns:
        Dict with results for pinned and pageable transfers
    """
    size_bytes = size_mb * 1024 * 1024

    results = {}

    # Pageable memory
    h2d_page, d2h_page = benchmark_memory_transfer(size_bytes, device, use_pinned=False, iterations=iterations)
    results['pageable'] = {
        'h2d_gbs': h2d_page,
        'd2h_gbs': d2h_page
    }

    # Pinned memory (only for CUDA)
    if device.type == 'cuda':
        h2d_pin, d2h_pin = benchmark_memory_transfer(size_bytes, device, use_pinned=True, iterations=iterations)
        results['pinned'] = {
            'h2d_gbs': h2d_pin,
            'd2h_gbs': d2h_pin
        }
        results['pinned_speedup'] = {
            'h2d': h2d_pin / h2d_page if h2d_page > 0 else 0,
            'd2h': d2h_pin / d2h_page if d2h_page > 0 else 0
        }

    return results
