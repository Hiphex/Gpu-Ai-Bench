"""
GPU Detection Module
Detects and identifies NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (MPS) GPUs
"""

import torch
import platform
from typing import Dict, Optional


class GPUDetector:
    """Detects and provides information about available GPUs"""

    def __init__(self):
        self.device = None
        self.gpu_type = None
        self.backend = None
        self.gpu_info = {}
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect available GPU and gather information"""
        # Check for MPS (Apple Silicon) first
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.backend = "MPS"
            self.gpu_type = "APPLE"
            self.gpu_info = self._gather_mps_info()
        # Check for CUDA (NVIDIA) or ROCm (AMD)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.backend = "CUDA" if torch.version.cuda else "ROCm"
            self.gpu_info = self._gather_cuda_info()
        else:
            raise RuntimeError(
                "No compatible GPU detected. Please ensure PyTorch is installed with GPU support:\n"
                "  - NVIDIA: Install with CUDA support\n"
                "  - AMD: Install with ROCm support\n"
                "  - Apple Silicon: Install PyTorch 2.0+ with MPS support"
            )

    def _gather_mps_info(self) -> Dict:
        """Gather Apple Silicon MPS GPU information"""
        import subprocess

        info = {
            "vendor": "Apple",
            "backend": "MPS",
            "platform": platform.system(),
            "pytorch_version": torch.__version__,
            "name": None,  # Will be set from device_name
            "device_name": None,
            "gpu_family": None,
            "total_memory_gb": None,
        }

        # Try to get system info using system_profiler on macOS
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout

            # Parse chip information
            for line in output.split('\n'):
                if 'Chipset Model:' in line or 'Graphics:' in line:
                    info["device_name"] = line.split(':', 1)[1].strip()
                elif 'Metal' in line and 'Family' in line:
                    info["gpu_family"] = line.split(':', 1)[1].strip()
        except:
            pass

        # Try to get CPU info which includes GPU on Apple Silicon
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            cpu_info = result.stdout.strip()
            if not info["device_name"]:
                info["device_name"] = cpu_info
        except:
            pass

        # Estimate memory (Apple Silicon uses unified memory)
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True,
                text=True,
                timeout=5
            )
            total_memory_bytes = int(result.stdout.strip())
            # Unified memory, typically ~75% available for GPU
            info["total_memory_gb"] = (total_memory_bytes / (1024**3)) * 0.75
        except:
            pass

        # Set defaults if detection failed
        if not info["device_name"]:
            info["device_name"] = "Apple Silicon GPU (MPS)"

        # Set name field for consistency with CUDA/ROCm
        info["name"] = info["device_name"]

        return info

    def _gather_cuda_info(self) -> Dict:
        """Gather detailed GPU information"""
        info = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": None,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "cuda_cores": None,
            "tensor_cores": None,
            "platform": platform.system(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.version.cuda else "N/A",
            "backend": self.backend
        }

        # Get compute capability
        props = torch.cuda.get_device_properties(0)
        info["compute_capability"] = f"{props.major}.{props.minor}"

        # Detect if it's NVIDIA or AMD
        device_name = info["name"].upper()
        if "NVIDIA" in device_name or "RTX" in device_name or "GTX" in device_name or "TESLA" in device_name:
            self.gpu_type = "NVIDIA"
            info["vendor"] = "NVIDIA"
            # Estimate CUDA cores (this is approximate based on SMs)
            info["cuda_cores"] = self._estimate_cuda_cores(props)
            info["tensor_cores"] = self._has_tensor_cores(props)
        elif "AMD" in device_name or "RADEON" in device_name or "MI" in device_name:
            self.gpu_type = "AMD"
            info["vendor"] = "AMD"
            info["compute_units"] = props.multi_processor_count
            info["stream_processors"] = self._estimate_stream_processors(props)
        else:
            self.gpu_type = "UNKNOWN"
            info["vendor"] = "UNKNOWN"

        # Try to get additional NVIDIA-specific info using nvidia-ml-py (pynvml)
        if self.gpu_type == "NVIDIA":
            try:
                # Try nvidia-ml-py first (recommended), fall back to pynvml
                try:
                    import pynvml
                except ImportError:
                    # If neither is available, skip NVML features
                    info["power_limit_w"] = None
                    info["memory_bandwidth_gbs"] = None
                    return info

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Get power info
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # Convert to Watts
                    info["power_limit_w"] = power_limit
                except:
                    info["power_limit_w"] = None

                # Get memory bandwidth from NVML
                try:
                    # Get memory clock and bus width to calculate bandwidth
                    mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    # Memory bandwidth (GB/s) = (Memory Clock MHz * Memory Bus Width bits / 8) / 1000
                    # Bus width varies by GPU, so we'll try to get it from known values
                    bandwidth_gbs = self._get_memory_bandwidth_nvml(props, mem_clock_mhz)
                    info["memory_bandwidth_gbs"] = bandwidth_gbs
                except:
                    info["memory_bandwidth_gbs"] = None

                pynvml.nvmlShutdown()
            except ImportError:
                info["power_limit_w"] = None
                info["memory_bandwidth_gbs"] = None
            except Exception as e:
                # Suppress warning if already shown by torch
                pass

        return info

    def _estimate_cuda_cores(self, props) -> Optional[int]:
        """Estimate CUDA cores based on architecture"""
        # This is a simplified estimation
        sm_count = props.multi_processor_count
        major, minor = props.major, props.minor

        # CUDA cores per SM varies by architecture
        cores_per_sm = {
            (3, 0): 192,  # Kepler
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (7, 0): 64,   # Volta
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,  # Ampere (GA10x)
            (8, 9): 128,  # Ada Lovelace
            (9, 0): 128,  # Hopper
        }

        cuda_cores_per_sm = cores_per_sm.get((major, minor), 128)  # Default to 128
        return sm_count * cuda_cores_per_sm

    def _estimate_stream_processors(self, props) -> Optional[int]:
        """Estimate stream processors for AMD GPUs"""
        # AMD typically has 64 stream processors per compute unit
        return props.multi_processor_count * 64

    def _has_tensor_cores(self, props) -> bool:
        """Check if GPU has Tensor Cores (Volta and newer)"""
        return props.major >= 7

    def _get_memory_bandwidth(self, props) -> Optional[float]:
        """Estimate memory bandwidth in GB/s"""
        # This is theoretical and varies by GPU model
        # For more accurate results, this should be measured
        return None

    def _get_memory_bandwidth_nvml(self, props, mem_clock_mhz: int) -> Optional[float]:
        """Calculate memory bandwidth from NVML memory clock and known bus widths"""
        # Memory bandwidth (GB/s) = (Memory Clock MHz * Memory Bus Width bits / 8) / 1000
        # Common bus widths by architecture:
        bus_width_bits = None

        # Get GPU name to determine bus width
        gpu_name = props.name.upper()

        # Tesla T4 has 256-bit bus width
        if "T4" in gpu_name:
            bus_width_bits = 256
        # A100 has 5120-bit HBM2
        elif "A100" in gpu_name:
            bus_width_bits = 5120
        # H100 has 5120-bit HBM3
        elif "H100" in gpu_name:
            bus_width_bits = 5120
        # V100 has 4096-bit HBM2
        elif "V100" in gpu_name:
            bus_width_bits = 4096
        # RTX 3090 has 384-bit bus
        elif "3090" in gpu_name:
            bus_width_bits = 384
        # RTX 4090 has 384-bit bus
        elif "4090" in gpu_name:
            bus_width_bits = 384
        # RTX 3080 has 320-bit bus
        elif "3080" in gpu_name:
            bus_width_bits = 320
        # RTX 4080 has 256-bit bus
        elif "4080" in gpu_name:
            bus_width_bits = 256
        # Default estimates by compute capability
        elif props.major == 7:  # Turing/Volta
            bus_width_bits = 256
        elif props.major == 8:  # Ampere
            bus_width_bits = 320
        elif props.major == 9:  # Hopper
            bus_width_bits = 5120

        if bus_width_bits:
            # Calculate bandwidth: (clock * width / 8) / 1000
            # Multiply by 2 for DDR (Double Data Rate)
            bandwidth_gbs = (mem_clock_mhz * bus_width_bits * 2 / 8) / 1000
            return bandwidth_gbs

        return None

    def get_device(self) -> torch.device:
        """Get the PyTorch device"""
        return self.device

    def get_info(self) -> Dict:
        """Get GPU information dictionary"""
        return self.gpu_info

    def print_info(self):
        """Print GPU information in a readable format"""
        print("\n" + "="*60)
        print("GPU INFORMATION")
        print("="*60)
        for key, value in self.gpu_info.items():
            formatted_key = key.replace("_", " ").title()
            print(f"{formatted_key:.<40} {value}")
        print("="*60 + "\n")
