"""
Timing utilities for cross-platform GPU benchmarking
Provides consistent timing across CUDA, ROCm, and MPS backends
"""

import torch
import time
from typing import Callable, Tuple, List
import numpy as np


class DeviceTimer:
    """Cross-platform GPU timer with event-based timing for CUDA/ROCm"""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_events = (device.type == 'cuda')

        if self.use_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        """Start timing"""
        if self.use_events:
            self.start_event.record()
        else:
            # MPS: use wall-clock time
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and synchronize"""
        if self.use_events:
            self.end_event.record()
            torch.cuda.synchronize()
        else:
            if self.device.type == 'mps':
                torch.mps.synchronize()
            self.end_time = time.perf_counter()

    def elapsed_time_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.use_events:
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (self.end_time - self.start_time) * 1000.0


def benchmark_operation(
    operation: Callable,
    warmup_iterations: int = 10,
    test_iterations: int = 100,
    device: torch.device = None,
    return_stats: bool = True
) -> dict:
    """
    Benchmark an operation with proper warmup and statistics

    Args:
        operation: Callable that performs the operation to benchmark
        warmup_iterations: Number of warmup iterations
        test_iterations: Number of test iterations
        device: Device to use for timing
        return_stats: If True, return median, p10, p90; else return mean

    Returns:
        dict with timing statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

    # Warmup
    for _ in range(warmup_iterations):
        operation()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    # Benchmark
    times_ms = []
    for _ in range(test_iterations):
        timer = DeviceTimer(device)
        with timer:
            operation()
        times_ms.append(timer.elapsed_time_ms())

    times_ms = np.array(times_ms)

    if return_stats:
        return {
            'median_ms': float(np.median(times_ms)),
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'p10_ms': float(np.percentile(times_ms, 10)),
            'p90_ms': float(np.percentile(times_ms, 90)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
        }
    else:
        return {
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
        }


class MatmulPrecisionContext:
    """Context manager for controlling matmul precision on NVIDIA GPUs"""

    def __init__(self, precision: str = "highest", device: torch.device = None):
        """
        Args:
            precision: "highest" (FP32), "high" (TF32), or "medium" (mixed)
            device: Device to check if precision control is applicable
        """
        self.precision = precision
        self.device = device
        self.original_precision = None
        self.applicable = (device is None or device.type == 'cuda') and hasattr(torch, 'set_float32_matmul_precision')

    def __enter__(self):
        if self.applicable:
            self.original_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision(self.precision)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.applicable and self.original_precision is not None:
            torch.set_float32_matmul_precision(self.original_precision)


def get_matmul_precision_info() -> dict:
    """Get current matmul precision settings"""
    info = {}

    if hasattr(torch, 'get_float32_matmul_precision'):
        info['float32_matmul_precision'] = torch.get_float32_matmul_precision()
    else:
        info['float32_matmul_precision'] = 'N/A'

    if hasattr(torch.backends, 'cudnn'):
        info['cudnn_enabled'] = torch.backends.cudnn.enabled
        info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        info['cudnn_deterministic'] = torch.backends.cudnn.deterministic

    if hasattr(torch.backends, 'cuda'):
        info['cuda_matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
        info['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32

    return info
