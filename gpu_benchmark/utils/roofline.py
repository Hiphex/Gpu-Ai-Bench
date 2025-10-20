"""
Roofline analysis utilities
Helps understand if operations are compute-bound or bandwidth-bound
"""

from typing import Dict, List, Tuple
import numpy as np


class RooflineAnalysis:
    """
    Roofline model analysis for GPU performance

    The roofline model plots achieved performance (GFLOP/s) against
    operational intensity (FLOP/byte) to show if a kernel is:
    - Memory-bound: Limited by memory bandwidth
    - Compute-bound: Limited by peak FLOP/s

    Roofs:
    - Horizontal roof: Peak FLOP/s
    - Sloped roof: Memory bandwidth × operational intensity
    """

    def __init__(self, peak_flops: float, peak_bandwidth_gbs: float):
        """
        Args:
            peak_flops: Peak GFLOP/s
            peak_bandwidth_gbs: Peak memory bandwidth in GB/s
        """
        self.peak_gflops = peak_flops
        self.peak_bandwidth_gbs = peak_bandwidth_gbs

        # Ridge point: where compute and memory roofs intersect
        # operational_intensity = peak_FLOPS / peak_bandwidth
        self.ridge_point = peak_flops / peak_bandwidth_gbs if peak_bandwidth_gbs > 0 else 0

    def analyze_kernel(
        self,
        gflops_achieved: float,
        bytes_accessed: float,
        flops_executed: float
    ) -> Dict:
        """
        Analyze a kernel's performance

        Args:
            gflops_achieved: Achieved GFLOP/s
            bytes_accessed: Total bytes read/written
            flops_executed: Total floating point operations

        Returns:
            Dict with analysis results
        """
        # Operational intensity (FLOP/byte)
        operational_intensity = flops_executed / bytes_accessed if bytes_accessed > 0 else 0

        # Theoretical performance based on roofline
        if operational_intensity < self.ridge_point:
            # Memory-bound
            theoretical_gflops = operational_intensity * self.peak_bandwidth_gbs
            bound_by = "memory"
        else:
            # Compute-bound
            theoretical_gflops = self.peak_gflops
            bound_by = "compute"

        # Efficiency
        efficiency = (gflops_achieved / theoretical_gflops * 100) if theoretical_gflops > 0 else 0

        return {
            "operational_intensity": operational_intensity,
            "achieved_gflops": gflops_achieved,
            "theoretical_gflops": theoretical_gflops,
            "efficiency_percent": efficiency,
            "bound_by": bound_by,
            "ridge_point": self.ridge_point,
            "below_ridge": operational_intensity < self.ridge_point,
        }

    def analyze_matmul(self, M: int, N: int, K: int, time_s: float, dtype_bytes: int = 4) -> Dict:
        """
        Analyze matrix multiplication performance

        Args:
            M, N, K: Matrix dimensions (M×K × K×N)
            time_s: Time taken in seconds
            dtype_bytes: Bytes per element (4 for FP32, 2 for FP16, etc.)

        Returns:
            Dict with roofline analysis
        """
        # FLOPs for matrix multiplication: 2*M*N*K
        flops = 2 * M * N * K
        gflops = (flops / time_s) / 1e9

        # Bytes accessed (read A, B; write C)
        bytes_accessed = (M * K + K * N + M * N) * dtype_bytes

        return self.analyze_kernel(gflops, bytes_accessed, flops)

    def generate_roofline_data(self, intensity_range: Tuple[float, float] = (0.1, 1000)) -> Dict:
        """
        Generate data for plotting roofline

        Args:
            intensity_range: Range of operational intensities to plot

        Returns:
            Dict with x/y coordinates for roofline plot
        """
        intensities = np.logspace(
            np.log10(intensity_range[0]),
            np.log10(intensity_range[1]),
            num=1000
        )

        # Memory-bound region (sloped)
        memory_bound_perf = intensities * self.peak_bandwidth_gbs
        memory_bound_perf = np.minimum(memory_bound_perf, self.peak_gflops)

        # Compute-bound region (flat)
        compute_bound_perf = np.full_like(intensities, self.peak_gflops)

        # Actual roofline (minimum of both)
        roofline_perf = np.minimum(memory_bound_perf, compute_bound_perf)

        return {
            "intensities": intensities.tolist(),
            "roofline_gflops": roofline_perf.tolist(),
            "memory_roof_gflops": memory_bound_perf.tolist(),
            "compute_roof_gflops": compute_bound_perf.tolist(),
            "ridge_point": self.ridge_point,
            "peak_gflops": self.peak_gflops,
            "peak_bandwidth_gbs": self.peak_bandwidth_gbs,
        }

    @staticmethod
    def estimate_peak_specs(gpu_name: str, vendor: str) -> Tuple[float, float]:
        """
        Estimate peak specs for known GPUs
        Returns (peak_gflops_fp32, peak_bandwidth_gbs)
        """
        # This is a simplified lookup - real values vary by boost clocks, etc.
        specs = {
            # NVIDIA
            "RTX 4090": (82580, 1008),
            "RTX 4080": (48740, 716),
            "RTX 3090": (35580, 936),
            "A100": (19500, 1935),  # PCIe, FP32
            "H100": (51000, 3350),  # PCIe, FP32

            # AMD
            "MI250X": (47900, 3277),
            "RX 7900 XTX": (61400, 960),

            # Apple Silicon (estimated FP32)
            "M1": (2600, 68),
            "M1 Pro": (5200, 200),
            "M1 Max": (10400, 400),
            "M2": (3600, 100),
            "M2 Pro": (6800, 200),
            "M2 Max": (13600, 400),
            "M3": (4000, 100),
            "M3 Pro": (10000, 273),  # 18 GB/s unified × 4 channels × 3.79 = ~273 GB/s
            "M3 Max": (14000, 400),
            "M4": (4500, 120),
        }

        # Try exact match
        for key, (gflops, bw) in specs.items():
            if key.lower() in gpu_name.lower():
                return (gflops, bw)

        # Fallback defaults by vendor
        if vendor == "NVIDIA":
            return (10000, 500)
        elif vendor == "AMD":
            return (10000, 500)
        elif vendor == "Apple":
            return (5000, 200)
        else:
            return (10000, 500)
