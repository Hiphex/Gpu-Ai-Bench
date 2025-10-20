"""
Power monitoring utilities for different GPU vendors
"""

import subprocess
import time
import threading
from typing import Optional, List, Dict
from abc import ABC, abstractmethod


class PowerMonitor(ABC):
    """Abstract base class for power monitoring"""

    def __init__(self):
        self.samples = []
        self.monitoring = False
        self.monitor_thread = None

    @abstractmethod
    def get_current_power(self) -> Optional[float]:
        """Get current power draw in Watts"""
        pass

    @abstractmethod
    def get_current_temperature(self) -> Optional[float]:
        """Get current temperature in Celsius"""
        pass

    def start_monitoring(self, interval_seconds: float = 0.1):
        """Start background power monitoring"""
        self.monitoring = True
        self.samples = []

        def monitor():
            while self.monitoring:
                power = self.get_current_power()
                temp = self.get_current_temperature()
                if power is not None:
                    self.samples.append({
                        'timestamp': time.time(),
                        'power_w': power,
                        'temperature_c': temp
                    })
                time.sleep(interval_seconds)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        if not self.samples:
            return {}

        powers = [s['power_w'] for s in self.samples if s['power_w'] is not None]
        temps = [s['temperature_c'] for s in self.samples if s['temperature_c'] is not None]

        result = {
            'sample_count': len(self.samples),
            'duration_seconds': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'] if len(self.samples) > 1 else 0,
        }

        if powers:
            result.update({
                'avg_power_w': sum(powers) / len(powers),
                'max_power_w': max(powers),
                'min_power_w': min(powers),
                'total_energy_j': sum(powers) * (result['duration_seconds'] / len(powers)) if result['duration_seconds'] > 0 else 0,
            })

        if temps:
            result.update({
                'avg_temperature_c': sum(temps) / len(temps),
                'max_temperature_c': max(temps),
            })

        return result


class NVIDIAPowerMonitor(PowerMonitor):
    """NVIDIA GPU power monitoring via pynvml"""

    def __init__(self, device_index: int = 0):
        super().__init__()
        self.device_index = device_index
        self.handle = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.pynvml = pynvml
            self.available = True
        except:
            self.available = False

    def get_current_power(self) -> Optional[float]:
        """Get current power draw in Watts"""
        if not self.available:
            return None
        try:
            power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power_mw / 1000.0
        except:
            return None

    def get_current_temperature(self) -> Optional[float]:
        """Get current temperature in Celsius"""
        if not self.available:
            return None
        try:
            return float(self.pynvml.nvmlDeviceGetTemperature(self.handle, 0))
        except:
            return None

    def __del__(self):
        if self.available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


class AMDPowerMonitor(PowerMonitor):
    """AMD GPU power monitoring via rocm-smi"""

    def __init__(self, device_index: int = 0):
        super().__init__()
        self.device_index = device_index
        self.available = self._check_rocm_smi()

    def _check_rocm_smi(self) -> bool:
        """Check if rocm-smi is available"""
        try:
            result = subprocess.run(['rocm-smi', '--showpower'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def get_current_power(self) -> Optional[float]:
        """Get current power draw in Watts"""
        if not self.available:
            return None
        try:
            result = subprocess.run(
                ['rocm-smi', f'--device', str(self.device_index), '--showpower'],
                capture_output=True,
                text=True,
                timeout=2
            )
            # Parse output for power value
            for line in result.stdout.split('\n'):
                if 'Average Graphics Package Power' in line or 'GPU Power' in line:
                    # Extract number (format: "123.4 W" or similar)
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'W' in part or (i > 0 and parts[i-1].replace('.', '').isdigit()):
                            try:
                                return float(parts[i-1] if 'W' in part else part)
                            except:
                                pass
            return None
        except:
            return None

    def get_current_temperature(self) -> Optional[float]:
        """Get current temperature in Celsius"""
        if not self.available:
            return None
        try:
            result = subprocess.run(
                ['rocm-smi', f'--device', str(self.device_index), '--showtemp'],
                capture_output=True,
                text=True,
                timeout=2
            )
            # Parse output for temperature
            for line in result.stdout.split('\n'):
                if 'Temperature' in line or 'Edge' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'C' in part or (i > 0 and parts[i-1].replace('.', '').isdigit()):
                            try:
                                return float(parts[i-1] if 'C' in part else part)
                            except:
                                pass
            return None
        except:
            return None


class ApplePowerMonitor(PowerMonitor):
    """Apple Silicon power monitoring via powermetrics"""

    def __init__(self):
        super().__init__()
        self.available = self._check_powermetrics()
        self.process = None

    def _check_powermetrics(self) -> bool:
        """Check if powermetrics is available (requires sudo)"""
        try:
            # Test if powermetrics exists
            result = subprocess.run(['which', 'powermetrics'], capture_output=True)
            return result.returncode == 0
        except:
            return False

    def get_current_power(self) -> Optional[float]:
        """Get current GPU power draw in Watts"""
        if not self.available:
            return None
        try:
            # Run powermetrics for a single sample
            # Note: This requires sudo, so it may not work without proper permissions
            result = subprocess.run(
                ['sudo', 'powermetrics', '--samplers', 'gpu_power', '-i', '100', '-n', '1'],
                capture_output=True,
                text=True,
                timeout=3
            )
            # Parse GPU power from output
            for line in result.stdout.split('\n'):
                if 'GPU Power' in line or 'ANE Power' in line:
                    # Format: "GPU Power: 1234 mW"
                    parts = line.split(':')
                    if len(parts) > 1:
                        power_str = parts[1].strip().split()[0]
                        try:
                            power_mw = float(power_str)
                            return power_mw / 1000.0  # Convert to Watts
                        except:
                            pass
            return None
        except:
            return None

    def get_current_temperature(self) -> Optional[float]:
        """Get current temperature - not easily available on Apple Silicon"""
        # Apple doesn't expose GPU temperature via powermetrics
        # Would need IOKit or other low-level APIs
        return None


def create_power_monitor(device_type: str, device_index: int = 0) -> Optional[PowerMonitor]:
    """Factory function to create appropriate power monitor"""
    if device_type == 'cuda':
        # Try NVIDIA first
        monitor = NVIDIAPowerMonitor(device_index)
        if monitor.available:
            return monitor

        # Try AMD ROCm
        monitor = AMDPowerMonitor(device_index)
        if monitor.available:
            return monitor

    elif device_type == 'mps':
        monitor = ApplePowerMonitor()
        if monitor.available:
            return monitor

    return None
