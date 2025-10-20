"""
Power Benchmark Module
Monitors power consumption during GPU workloads
"""

import torch
import time
from typing import Dict, Optional


class PowerBenchmark:
    """Benchmark GPU power consumption"""

    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}
        self.nvml_available = False
        self.handle = None

        # Try to initialize NVML for NVIDIA GPUs
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_available = True
            self.pynvml = pynvml
        except ImportError:
            print("Warning: pynvml not available. Install with: pip install pynvml")
        except Exception as e:
            print(f"Warning: Could not initialize NVML: {e}")

    def __del__(self):
        """Cleanup NVML"""
        if self.nvml_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass

    def run_all(self) -> Dict:
        """Run all power benchmarks"""
        print("\n" + "="*60)
        print("POWER BENCHMARK")
        print("="*60)

        if not self.nvml_available:
            print("Power monitoring not available (NVIDIA Management Library not found)")
            print("This feature requires NVIDIA GPU with NVML support")
            return {"available": False, "message": "NVML not available"}

        # Get power information
        self.results["power_info"] = self._get_power_info()

        # Monitor power during stress test
        self.results["stress_test"] = self._power_stress_test()

        return self.results

    def _get_power_info(self) -> Dict:
        """Get GPU power information"""
        info = {}

        try:
            # Power limit
            power_limit = self.pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
            info["power_limit_w"] = power_limit

            # Current power draw
            power_draw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            info["current_power_w"] = power_draw

            # Temperature
            temperature = self.pynvml.nvmlDeviceGetTemperature(self.handle, 0)
            info["temperature_c"] = temperature

            # Fan speed (if available)
            try:
                fan_speed = self.pynvml.nvmlDeviceGetFanSpeed(self.handle)
                info["fan_speed_percent"] = fan_speed
            except:
                info["fan_speed_percent"] = None

            print(f"\nPower Information:")
            print(f"  Power Limit: {info['power_limit_w']:.2f} W")
            print(f"  Current Power: {info['current_power_w']:.2f} W")
            print(f"  Temperature: {info['temperature_c']} °C")
            if info['fan_speed_percent'] is not None:
                print(f"  Fan Speed: {info['fan_speed_percent']}%")

        except Exception as e:
            print(f"Error getting power info: {e}")
            info["error"] = str(e)

        return info

    def _power_stress_test(self, duration_seconds: float = 10.0) -> Dict:
        """Run a stress test and monitor power consumption"""
        print(f"\nRunning power stress test for {duration_seconds} seconds...")

        power_samples = []
        temp_samples = []

        # Create a heavy workload
        matrix_size = 8192
        A = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=self.device)
        B = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=self.device)

        start_time = time.time()
        sample_count = 0

        try:
            while time.time() - start_time < duration_seconds:
                # Heavy computation
                C = torch.matmul(A, B)
                torch.cuda.synchronize()

                # Sample power every ~0.5 seconds
                if sample_count % 10 == 0:
                    try:
                        power = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                        temp = self.pynvml.nvmlDeviceGetTemperature(self.handle, 0)
                        power_samples.append(power)
                        temp_samples.append(temp)
                    except:
                        pass

                sample_count += 1

        except KeyboardInterrupt:
            print("Stress test interrupted")

        finally:
            del A, B, C
            torch.cuda.empty_cache()

        # Calculate statistics
        if power_samples:
            results = {
                "duration_seconds": duration_seconds,
                "avg_power_w": sum(power_samples) / len(power_samples),
                "max_power_w": max(power_samples),
                "min_power_w": min(power_samples),
                "avg_temperature_c": sum(temp_samples) / len(temp_samples),
                "max_temperature_c": max(temp_samples),
                "samples": len(power_samples)
            }

            print(f"  Average Power: {results['avg_power_w']:.2f} W")
            print(f"  Max Power: {results['max_power_w']:.2f} W")
            print(f"  Average Temperature: {results['avg_temperature_c']:.1f} °C")
            print(f"  Max Temperature: {results['max_temperature_c']} °C")

            return results
        else:
            return {"error": "No power samples collected"}

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print a summary of power results"""
        print("\n" + "="*60)
        print("POWER BENCHMARK SUMMARY")
        print("="*60)

        if not self.results.get("available", True):
            print("Power monitoring not available")
        else:
            power_info = self.results.get("power_info", {})
            stress_test = self.results.get("stress_test", {})

            print(f"Power Limit:................ {power_info.get('power_limit_w', 0):.2f} W")
            print(f"Idle Power:................. {power_info.get('current_power_w', 0):.2f} W")
            print(f"Stress Test Avg Power:...... {stress_test.get('avg_power_w', 0):.2f} W")
            print(f"Stress Test Max Power:...... {stress_test.get('max_power_w', 0):.2f} W")

        print("="*60)
