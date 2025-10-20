"""
Thermal monitoring and throttling detection utilities
"""

import time
import torch
from typing import Dict, List, Optional
import warnings


class ThermalMonitor:
    """
    Monitor GPU temperature and detect thermal throttling

    Thermal throttling occurs when GPU reduces clock speeds due to high temperature
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.backend = device.type

        # Initialize backend-specific monitoring
        if self.backend == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                self.pynvml = pynvml
                self.nvidia_available = True
            except:
                self.nvidia_available = False
                warnings.warn("pynvml not available - NVIDIA thermal monitoring disabled")
        else:
            self.nvidia_available = False

    def get_temperature(self) -> Optional[float]:
        """
        Get current GPU temperature in Celsius

        Returns:
            Temperature in °C, or None if not available
        """
        if self.backend == 'cuda' and self.nvidia_available:
            try:
                temp = self.pynvml.nvmlDeviceGetTemperature(
                    self.handle,
                    self.pynvml.NVML_TEMPERATURE_GPU
                )
                return float(temp)
            except:
                return None

        elif self.backend == 'mps':
            # Apple Silicon temperature monitoring via powermetrics requires sudo
            # Not implemented here - would need subprocess call to powermetrics
            return None

        return None

    def get_clock_speeds(self) -> Optional[Dict]:
        """
        Get current GPU clock speeds

        Returns:
            Dict with graphics_clock_mhz and memory_clock_mhz, or None
        """
        if self.backend == 'cuda' and self.nvidia_available:
            try:
                graphics_clock = self.pynvml.nvmlDeviceGetClockInfo(
                    self.handle,
                    self.pynvml.NVML_CLOCK_GRAPHICS
                )
                memory_clock = self.pynvml.nvmlDeviceGetClockInfo(
                    self.handle,
                    self.pynvml.NVML_CLOCK_MEM
                )
                return {
                    "graphics_clock_mhz": graphics_clock,
                    "memory_clock_mhz": memory_clock
                }
            except:
                return None

        return None

    def get_throttle_reasons(self) -> Optional[Dict]:
        """
        Get current throttling reasons (NVIDIA only)

        Returns:
            Dict with throttle reasons, or None
        """
        if self.backend == 'cuda' and self.nvidia_available:
            try:
                reasons = self.pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)

                return {
                    "gpu_idle": bool(reasons & self.pynvml.nvmlClocksThrottleReasonGpuIdle),
                    "thermal_limit": bool(reasons & self.pynvml.nvmlClocksThrottleReasonSwThermalSlowdown),
                    "power_limit": bool(reasons & self.pynvml.nvmlClocksThrottleReasonSwPowerCap),
                    "hw_thermal_limit": bool(reasons & self.pynvml.nvmlClocksThrottleReasonHwThermalSlowdown),
                    "hw_power_limit": bool(reasons & self.pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown),
                }
            except:
                return None

        return None

    def monitor_during_benchmark(self, duration_sec: float = 10.0, sample_interval: float = 0.5) -> Dict:
        """
        Monitor temperature and clocks during a benchmark

        Args:
            duration_sec: How long to monitor (seconds)
            sample_interval: Time between samples (seconds)

        Returns:
            Dict with temperature history, clock speeds, and throttle detection
        """
        temperatures = []
        clock_speeds = []
        throttle_events = []

        start_time = time.time()
        sample_count = 0

        while (time.time() - start_time) < duration_sec:
            # Get temperature
            temp = self.get_temperature()
            if temp is not None:
                temperatures.append(temp)

            # Get clock speeds
            clocks = self.get_clock_speeds()
            if clocks is not None:
                clock_speeds.append(clocks)

            # Check throttling
            throttle = self.get_throttle_reasons()
            if throttle is not None:
                if throttle['thermal_limit'] or throttle['hw_thermal_limit']:
                    throttle_events.append({
                        'time': time.time() - start_time,
                        'reasons': throttle
                    })

            sample_count += 1
            time.sleep(sample_interval)

        # Analysis
        results = {
            "samples": sample_count,
            "duration_sec": time.time() - start_time,
        }

        if temperatures:
            results["temperature"] = {
                "avg_celsius": sum(temperatures) / len(temperatures),
                "min_celsius": min(temperatures),
                "max_celsius": max(temperatures),
                "history": temperatures
            }

        if clock_speeds:
            avg_graphics = sum(c['graphics_clock_mhz'] for c in clock_speeds) / len(clock_speeds)
            avg_memory = sum(c['memory_clock_mhz'] for c in clock_speeds) / len(clock_speeds)

            results["clock_speeds"] = {
                "avg_graphics_mhz": avg_graphics,
                "avg_memory_mhz": avg_memory,
                "min_graphics_mhz": min(c['graphics_clock_mhz'] for c in clock_speeds),
                "max_graphics_mhz": max(c['graphics_clock_mhz'] for c in clock_speeds),
            }

        if throttle_events:
            results["throttling_detected"] = True
            results["throttle_events"] = throttle_events
        else:
            results["throttling_detected"] = False

        return results

    def get_thermal_headroom(self) -> Optional[Dict]:
        """
        Get thermal headroom information

        Returns:
            Dict with current temp, max temp, and headroom percentage
        """
        if self.backend == 'cuda' and self.nvidia_available:
            try:
                current_temp = self.pynvml.nvmlDeviceGetTemperature(
                    self.handle,
                    self.pynvml.NVML_TEMPERATURE_GPU
                )

                # Get temperature thresholds
                try:
                    shutdown_temp = self.pynvml.nvmlDeviceGetTemperatureThreshold(
                        self.handle,
                        self.pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
                    )
                except:
                    shutdown_temp = 95  # Default estimate

                try:
                    slowdown_temp = self.pynvml.nvmlDeviceGetTemperatureThreshold(
                        self.handle,
                        self.pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
                    )
                except:
                    slowdown_temp = 85  # Default estimate

                headroom = slowdown_temp - current_temp
                headroom_pct = (headroom / slowdown_temp) * 100

                return {
                    "current_temp_celsius": current_temp,
                    "slowdown_temp_celsius": slowdown_temp,
                    "shutdown_temp_celsius": shutdown_temp,
                    "headroom_celsius": headroom,
                    "headroom_percent": headroom_pct
                }
            except:
                return None

        return None

    def __del__(self):
        """Cleanup"""
        if self.backend == 'cuda' and self.nvidia_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    monitor = ThermalMonitor(device)

    print(f"Device: {device}")
    print(f"Temperature: {monitor.get_temperature()}°C")
    print(f"Clock speeds: {monitor.get_clock_speeds()}")
    print(f"Throttle reasons: {monitor.get_throttle_reasons()}")
    print(f"Thermal headroom: {monitor.get_thermal_headroom()}")
