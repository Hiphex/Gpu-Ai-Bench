"""
Benchmark modules for GPU testing
"""

from .flops_benchmark import FLOPSBenchmark
from .memory_benchmark import MemoryBenchmark
from .power_benchmark import PowerBenchmark
from .workload_benchmark import WorkloadBenchmark

__all__ = ['FLOPSBenchmark', 'MemoryBenchmark', 'PowerBenchmark', 'WorkloadBenchmark']
