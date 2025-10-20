<div align="center">

# 🚀 GPU AI Benchmark

**Production-Grade GPU Benchmarking for AI Workloads**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-NVIDIA%20|%20AMD%20|%20Apple-green.svg)]()

*Comprehensive GPU performance testing with precision control, roofline analysis, and multi-vendor support*

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Examples](#-examples)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Supported Hardware](#-supported-hardware)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
  - [Basic Benchmarks](#basic-benchmarks)
  - [Advanced Features](#advanced-features)
  - [Export Options](#export-options)
- [Benchmarks Explained](#-benchmarks-explained)
- [Advanced Features](#-advanced-features-guide)
  - [Precision Control](#1-precision-control)
  - [Enhanced Timing](#2-enhanced-timing)
  - [Memory Analysis](#3-memory-bandwidth-analysis)
  - [Power Monitoring](#4-multi-vendor-power-monitoring)
  - [Roofline Analysis](#5-roofline-analysis)
  - [SDPA Testing](#6-sdpa-backend-testing)
- [Performance Tips](#-performance-tips)
- [Developer API](#-developer-api)
- [Understanding Results](#-understanding-results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)

---

## 🎯 Overview

GPU AI Benchmark is a comprehensive, production-grade benchmarking tool designed specifically for evaluating GPU performance on AI/ML workloads. Unlike synthetic benchmarks, this tool focuses on real-world AI operations with proper timing, statistical analysis, and cross-platform support.

### Why This Benchmark?

- **Accurate Timing**: GPU event-based timing (CUDA/ROCm) with nanosecond precision
- **Statistical Rigor**: Median, p10/p90, std dev - not just averages
- **Precision Control**: Explicit FP32/TF32/BF16/FP16/FP8 testing
- **Real AI Workloads**: Convolutions, attention mechanisms, actual operations
- **Roofline Analysis**: Understand compute vs memory bottlenecks
- **Cross-Platform**: Works on NVIDIA, AMD, and Apple Silicon

---

## ✨ Features

<table>
<tr>
<td width="50%">

### Core Capabilities
- ⚡ **Multi-Platform Support**
  - NVIDIA CUDA GPUs
  - AMD ROCm GPUs
  - Apple Silicon (M1/M2/M3/M4)

- 📊 **Precision Testing**
  - FP32 (true floating point)
  - TF32 (NVIDIA Ampere+)
  - BF16 (Brain Float 16)
  - FP16 (half precision)
  - FP8 (8-bit, Hopper+)

- ⏱️ **Advanced Timing**
  - GPU event-based (CUDA/ROCm)
  - Statistical analysis
  - Median, p10, p90, std dev

</td>
<td width="50%">

### Advanced Features
- 💾 **Memory Analysis**
  - Pinned vs pageable transfers
  - D2D, H2D, D2H bandwidth
  - Unified memory support (MPS)

- ⚡ **Power Monitoring**
  - NVIDIA (pynvml)
  - AMD (rocm-smi)
  - Apple (powermetrics)
  - Energy efficiency (perf/W)

- 📈 **Roofline Analysis**
  - Compute vs memory bound
  - Operational intensity
  - Efficiency percentage

</td>
</tr>
</table>

### AI Workload Testing
- **SDPA Backends**: FlashAttention, Memory-Efficient, Math
- **Convolutions**: CNN operations for computer vision
- **Attention**: Transformer operations for NLP
- **Element-wise**: ReLU, GELU, Sigmoid, Tanh
- **Reductions**: Sum, Mean, Max, ArgMax

### Export & Reporting
- **JSON**: Complete results with metadata
- **CSV**: Flattened for easy comparison
- **Comprehensive Reports**: GPU info, settings, performance

---

## 💻 Supported Hardware

### NVIDIA GPUs (CUDA)
✅ GeForce RTX 20/30/40 series
✅ Tesla/A100/H100 datacenter GPUs
✅ Quadro workstation GPUs
**Requirements**: CUDA 11.0+, Compute Capability 6.0+

### AMD GPUs (ROCm)
✅ Radeon RX 6000/7000 series
✅ Radeon Pro workstation GPUs
✅ MI100/MI200 datacenter GPUs
**Requirements**: ROCm 5.0+

### Apple Silicon
✅ M1, M1 Pro, M1 Max, M1 Ultra
✅ M2, M2 Pro, M2 Max, M2 Ultra
✅ M3, M3 Pro, M3 Max
✅ M4, M4 Pro, M4 Max
**Requirements**: macOS 12.3+, PyTorch 2.0+

---

## 📦 Installation

### Prerequisites
- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ with GPU support
- **Platform-specific drivers**:
  - NVIDIA: CUDA drivers
  - AMD: ROCm drivers
  - Apple: macOS 12.3+

### Step 1: Clone Repository

```bash
git clone https://github.com/Hiphex/Gpu-Ai-Bench.git
cd Gpu-Ai-Bench
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install PyTorch with GPU Support

**NVIDIA CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**AMD ROCm:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Apple Silicon:**
```bash
pip install torch torchvision torchaudio
```

### Step 4: Optional - Power Monitoring

**NVIDIA:**
```bash
pip install pynvml
```

**AMD:**
```bash
# Ensure ROCm is installed
rocm-smi --version
```

**Apple:**
```bash
# Requires sudo permissions for powermetrics
# No additional installation needed
```

### Verification

```bash
# Verify GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()}')"
```

---

## 🚀 Quick Start

### Run Complete Benchmark

```bash
python benchmark.py --all
```

### Run with Export

```bash
python benchmark.py --all --json results.json --csv results.csv
```

### Run Specific Tests

```bash
# FLOPS only
python benchmark.py --flops

# Memory bandwidth
python benchmark.py --memory

# Power consumption (NVIDIA only)
python benchmark.py --power

# AI workloads
python benchmark.py --workload
```

---

## 📖 Usage Guide

### Basic Benchmarks

#### FLOPS Testing
```bash
# Test FP32, FP16, FP8
python benchmark.py --flops

# Custom iterations
python benchmark.py --flops --warmup 20 --iterations 200
```

#### Memory Bandwidth
```bash
# Test D2D, H2D, D2H
python benchmark.py --memory

# With custom output directory
python benchmark.py --memory --output-dir ./my_results
```

#### Power Monitoring
```bash
# NVIDIA GPUs only
python benchmark.py --power
```

#### AI Workloads
```bash
# Convolutions, attention, element-wise ops
python benchmark.py --workload
```

### Advanced Features

#### Using Enhanced FLOPS Benchmark (with TF32, BF16)
```python
from gpu_benchmark.benchmarks.flops_benchmark_v2 import EnhancedFLOPSBenchmark
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
bench = EnhancedFLOPSBenchmark(device, warmup_iterations=10, test_iterations=100)

# Run with all precisions including TF32
results = bench.run_all(include_tf32=True, include_bf16=True)
bench.print_summary()
```

#### SDPA Backend Testing
```python
from gpu_benchmark.benchmarks.attention_benchmark import SDPABenchmark

device = torch.device('cuda')
sdpa_bench = SDPABenchmark(device)
results = sdpa_bench.run_all()
sdpa_bench.print_summary()
```

#### Roofline Analysis
```python
from gpu_benchmark.utils.roofline import RooflineAnalysis

# Initialize with your GPU specs
roofline = RooflineAnalysis(
    peak_flops=82580,       # GFLOPS (e.g., RTX 4090 FP32)
    peak_bandwidth_gbs=1008  # GB/s
)

# Analyze a matrix multiplication
analysis = roofline.analyze_matmul(
    M=4096, N=4096, K=4096,
    time_s=0.025,
    dtype_bytes=4  # FP32 = 4 bytes
)

print(f"Operational Intensity: {analysis['operational_intensity']:.1f} FLOP/byte")
print(f"Bound by: {analysis['bound_by']}")
print(f"Efficiency: {analysis['efficiency_percent']:.1f}%")
```

### Export Options

#### JSON Export (Full Details)
```bash
python benchmark.py --all --json full_results.json
```

**Output structure:**
```json
{
  "timestamp": "2025-10-20T18:30:00",
  "gpu_info": {
    "name": "Apple M3 Pro",
    "vendor": "Apple",
    "backend": "MPS",
    "total_memory_gb": 18.0
  },
  "flops": {
    "fp32": {
      "median_tflops": 4.12,
      "mean_tflops": 4.08,
      "detailed_results": [...]
    }
  },
  "memory": {...},
  "power": {...},
  "workloads": {...}
}
```

#### CSV Export (Easy Comparison)
```bash
python benchmark.py --all --csv results.csv
```

---

## 🔬 Benchmarks Explained

### 1. FLOPS Benchmark
**What it measures**: Theoretical peak performance in TFLOPS (Trillions of Floating Point Operations Per Second)

**Precisions tested**:
- **FP32**: Standard single precision (baseline)
- **TF32**: TensorFloat-32 (NVIDIA Ampere+, ~8x faster than FP32)
- **BF16**: Brain Float 16 (good for training)
- **FP16**: Half precision (maximum throughput)
- **FP8**: 8-bit floating point (NVIDIA Hopper+)

**Test method**: Matrix multiplication (GEMM) at various sizes
- Small: 4096×4096
- Large: 8192×8192

**Output metrics**:
- Median TFLOPS (most reliable)
- Mean TFLOPS (average)
- p10/p90 range (variability)
- Time per operation

### 2. Memory Bandwidth
**What it measures**: Data transfer speeds in GB/s

**Transfer types**:
- **D2D**: Device-to-Device (internal GPU memory)
- **H2D**: Host-to-Device (CPU → GPU)
- **D2H**: Device-to-Host (GPU → CPU)

**Advanced**: Compares pinned vs pageable memory (CUDA only)
- Pinned: 2-3x faster but locks system RAM
- Pageable: Slower but more flexible

**Test sizes**: 128MB, 512MB, 1024MB, 2048MB

### 3. Power Monitoring
**What it measures**: Power consumption and thermal characteristics

**NVIDIA (via pynvml)**:
- Power draw (Watts)
- Power limit
- Temperature (°C)
- Fan speed (%)
- Energy consumed (Joules)

**AMD (via rocm-smi)**:
- Average package power
- GPU temperature
- Clock speeds

**Apple (via powermetrics)**:
- GPU power (requires sudo)
- ANE power
- Temperature (limited)

### 4. AI Workload Testing
**What it measures**: Real-world AI operation performance

**Operations tested**:

1. **Convolution** (CNN operations)
   - ResNet-like layers
   - Various batch sizes and channels
   - Throughput in images/second

2. **Attention** (Transformer operations)
   - Multi-head attention
   - BERT-like and large context configs
   - Throughput in tokens/second

3. **Element-wise** (Activations)
   - ReLU, GELU, Sigmoid, Tanh
   - Throughput in GB/s

4. **Reduction** (Aggregations)
   - Sum, Mean, Max, ArgMax
   - Throughput in GB/s

---

## 🎓 Advanced Features Guide

### 1. Precision Control

#### TF32 vs FP32 on NVIDIA GPUs

**Background**: NVIDIA Ampere and newer GPUs support TensorFloat-32 (TF32), which provides ~8x speedup over true FP32 with minimal accuracy loss (8-bit mantissa vs 23-bit).

**Usage**:
```python
from gpu_benchmark.utils import MatmulPrecisionContext

# True FP32 (highest precision)
with MatmulPrecisionContext("highest", device):
    result = torch.matmul(A, B)

# TF32 (faster, still accurate)
with MatmulPrecisionContext("high", device):
    result = torch.matmul(A, B)
```

**When to use**:
- **FP32**: Scientific computing, high precision required
- **TF32**: AI training/inference, 8x faster with <0.1% accuracy loss
- **FP16**: Maximum throughput, some accuracy trade-off
- **BF16**: Training stability, better than FP16 for gradients
- **FP8**: Inference only, maximum speed on Hopper+

### 2. Enhanced Timing

**GPU Event-Based Timing** (CUDA/ROCm):
```python
from gpu_benchmark.utils import DeviceTimer

timer = DeviceTimer(device)
with timer:
    result = torch.matmul(A, B)
elapsed_ms = timer.elapsed_time_ms()
```

**Benefits**:
- Nanosecond precision from GPU hardware
- Eliminates CPU-GPU synchronization overhead
- More accurate than `time.perf_counter()`

**Statistical Analysis**:
```python
from gpu_benchmark.utils import benchmark_operation

def my_operation():
    return torch.matmul(A, B)

stats = benchmark_operation(
    my_operation,
    warmup_iterations=10,
    test_iterations=100,
    device=device,
    return_stats=True
)

# Returns: median, mean, std, p10, p90, min, max
print(f"Median: {stats['median_ms']:.2f} ms")
print(f"Range: [{stats['p10_ms']:.2f} - {stats['p90_ms']:.2f}] ms")
```

### 3. Memory Bandwidth Analysis

#### Pinned vs Pageable Memory

**Pageable (default)**:
```python
data = torch.randn(size, device='cpu')
# Slower transfers, but doesn't lock RAM
```

**Pinned (CUDA only)**:
```python
data = torch.randn(size, device='cpu', pin_memory=True)
# 2-3x faster H2D/D2H, but locks RAM
```

**Benchmark comparison**:
```python
from gpu_benchmark.utils.memory import compare_pinned_vs_pageable

results = compare_pinned_vs_pageable(
    size_mb=512,
    device=device,
    iterations=50
)

print(f"Pageable H2D: {results['pageable']['h2d_gbs']:.2f} GB/s")
print(f"Pinned H2D: {results['pinned']['h2d_gbs']:.2f} GB/s")
print(f"Speedup: {results['pinned_speedup']['h2d']:.2f}x")
```

### 4. Multi-Vendor Power Monitoring

```python
from gpu_benchmark.utils.power import create_power_monitor

# Auto-detects vendor
monitor = create_power_monitor(device.type)

# Start background monitoring
monitor.start_monitoring(interval_seconds=0.1)

# ... run your workload ...

# Stop and get statistics
stats = monitor.stop_monitoring()

print(f"Average Power: {stats['avg_power_w']:.1f} W")
print(f"Max Power: {stats['max_power_w']:.1f} W")
print(f"Total Energy: {stats['total_energy_j']:.1f} J")
print(f"Avg Temperature: {stats['avg_temperature_c']:.1f} °C")
```

**Platform-specific notes**:
- **NVIDIA**: Works out of the box with `pynvml`
- **AMD**: Requires ROCm installation
- **Apple**: Requires `sudo` permissions for `powermetrics`

### 5. Roofline Analysis

**The Roofline Model** helps understand if your operations are:
- **Compute-bound**: Limited by FLOP/s (above ridge point)
- **Memory-bound**: Limited by bandwidth (below ridge point)

**Key Concepts**:
- **Operational Intensity** = FLOPs / Bytes Accessed
- **Ridge Point** = Peak FLOP/s / Peak Bandwidth

**Example**:
```python
from gpu_benchmark.utils.roofline import RooflineAnalysis

# Use your GPU's specs
roofline = RooflineAnalysis(
    peak_flops=10000,    # GFLOP/s
    peak_bandwidth_gbs=273  # GB/s (e.g., M3 Pro)
)

# Analyze a matmul
analysis = roofline.analyze_matmul(
    M=4096, N=4096, K=4096,
    time_s=0.028,
    dtype_bytes=4
)

print(f"Operational Intensity: {analysis['operational_intensity']:.1f} FLOP/byte")
print(f"Ridge Point: {analysis['ridge_point']:.1f} FLOP/byte")
print(f"→ {analysis['bound_by'].upper()}-BOUND")
print(f"Efficiency: {analysis['efficiency_percent']:.1f}% of theoretical peak")
```

**Interpretation**:
- **Below ridge** (memory-bound): Improve data reuse, use caching
- **Above ridge** (compute-bound): Optimize algorithm, use lower precision

### 6. SDPA Backend Testing

PyTorch 2.0+ supports multiple Scaled Dot Product Attention backends:

```python
from gpu_benchmark.benchmarks.attention_benchmark import SDPABenchmark

bench = SDPABenchmark(device)
results = bench.run_all()
```

**Backends tested**:
1. **FlashAttention**: Fastest, requires Ampere+ (compute capability 8.0+)
2. **Memory-Efficient**: Slower but more compatible
3. **Math (Baseline)**: Pure PyTorch, slowest but works everywhere

**Example output**:
```
BERT-like (batch=16, seq=512):
  FlashAttention............... 12.34 ms (661,504 tokens/s)
  Memory-Efficient............. 18.76 ms (435,200 tokens/s)
  Math (Baseline).............. 45.23 ms (180,480 tokens/s)
```

---

## 💡 Performance Tips

### General Optimization
1. **Close Background Apps**: Free up GPU for accurate benchmarks
2. **Thermal Management**: Ensure adequate cooling (fans, thermal paste)
3. **Power Mode**: Use "High Performance" power plan
4. **Multiple Runs**: Run 2-3 times, use median results
5. **Driver Updates**: Keep GPU drivers up to date

### Platform-Specific Tips

#### NVIDIA GPUs
- **TF32**: Use for 8x speedup with minimal accuracy loss
- **Tensor Cores**: Ensure FP16/BF16 to utilize them fully
- **CUDA Graphs**: For repeated workloads (advanced)
- **cuDNN**: Enable `torch.backends.cudnn.benchmark = True`

#### AMD GPUs
- **ROCm Version**: Use latest ROCm for best performance
- **HSA**: Ensure HSA_OVERRIDE_GFX_VERSION is set correctly
- **Memory**: AMD often has more VRAM, use larger batches

#### Apple Silicon
- **Plug In**: Performance drops 40-60% on battery
- **Free RAM**: Unified memory shared with system
- **Low Power Mode**: Disable in System Preferences
- **Thermal Stabilization**: First run may be slower
- **Expected Performance**:
  ```
  M1/M2:         3-6 TFLOPS FP32,  10-20 TFLOPS FP16
  M1/M2 Pro/Max: 5-10 TFLOPS FP32, 15-30 TFLOPS FP16
  M3 Pro/Max:    6-12 TFLOPS FP32, 20-40 TFLOPS FP16
  M4:            8-15 TFLOPS FP32, 25-50 TFLOPS FP16
  ```

### Precision Selection
- **FP32**: Scientific computing, maximum accuracy
- **TF32**: Default for NVIDIA Ampere+ AI workloads
- **BF16**: Training stability (better than FP16 for gradients)
- **FP16**: Maximum throughput for inference
- **FP8**: Cutting-edge inference (Hopper+ only)

---

## 👨‍💻 Developer API

All utilities are importable for custom benchmarks:

```python
from gpu_benchmark.utils import (
    DeviceTimer,
    benchmark_operation,
    MatmulPrecisionContext,
    get_matmul_precision_info
)
from gpu_benchmark.utils.roofline import RooflineAnalysis
from gpu_benchmark.utils.power import create_power_monitor
from gpu_benchmark.utils.memory import (
    benchmark_memory_transfer,
    compare_pinned_vs_pageable
)

# Example: Custom benchmark with proper timing
def my_custom_benchmark(device):
    A = torch.randn(4096, 4096, device=device)
    B = torch.randn(4096, 4096, device=device)

    stats = benchmark_operation(
        lambda: torch.matmul(A, B),
        warmup_iterations=10,
        test_iterations=100,
        device=device
    )

    return stats['median_ms']

# Example: Roofline analysis of custom kernel
roofline = RooflineAnalysis(peak_flops=10000, peak_bandwidth_gbs=500)
analysis = roofline.analyze_kernel(
    gflops_achieved=8500,
    bytes_accessed=1024**3,  # 1 GB
    flops_executed=8500 * 10**9
)
```

---

## 📊 Understanding Results

### Example Output

```
============================================================
GPU INFORMATION
============================================================
Name........................ Apple M3 Pro
Vendor...................... Apple
Backend..................... MPS
Total Memory................ 18.0 GB
Platform.................... Darwin
============================================================

============================================================
ENHANCED FLOPS BENCHMARK
============================================================
Precision Settings:
  float32_matmul_precision: highest
  cudnn_allow_tf32: False

Benchmarking FP32 (True) Matrix Multiplication
Matrix size: 4096x4096 × 4096x4096
  Median: 4.12 TFLOPS (26.50 ms)
  Mean:   4.08 TFLOPS (26.75 ms)
  Std:    0.15 ms
  Range:  [26.35 - 27.10] ms (p10-p90)

Matrix size: 8192x8192 × 8192x8192
  Median: 3.48 TFLOPS (315.20 ms)

Benchmarking FP16 Matrix Multiplication
  Median: 4.27 TFLOPS (25.55 ms)

============================================================
MEMORY BENCHMARK
============================================================
Total Memory: 13.5 GB
Peak Bandwidth: 57.58 GB/s

============================================================
AI WORKLOAD BENCHMARK
============================================================
Convolution Throughput: 4,745 img/s
Attention Throughput: 527,254 tokens/s
```

### Interpreting Metrics

**TFLOPS**:
- Higher is better
- Compare across precisions (FP16 should be ~2x FP32)
- Compare with theoretical peak (efficiency)

**Memory Bandwidth**:
- D2D should be highest (internal bandwidth)
- H2D/D2H limited by PCIe/system bus
- Pinned memory shows speedup potential

**Power**:
- Avg Power: Sustained draw during workload
- Max Power: Peak consumption
- Energy: Total joules consumed
- **Efficiency**: TFLOPS / Watt

**Roofline**:
- Above ridge: Compute-bound (good for GEMM)
- Below ridge: Memory-bound (improve reuse)
- Efficiency %: How close to theoretical max

---

## 🔧 Troubleshooting

### GPU Not Detected

**Error**: `No compatible GPU detected`

**NVIDIA/AMD**:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check device count
python -c "import torch; print(torch.cuda.device_count())"

# Reinstall PyTorch with GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon**:
```bash
# Check MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# Requirements
# - macOS 12.3+
# - PyTorch 2.0+
pip install --upgrade torch torchvision torchaudio
```

### Out of Memory

**Error**: `RuntimeError: CUDA/MPS out of memory`

**Solutions**:
1. **Reduce iterations**:
   ```bash
   python benchmark.py --all --iterations 50
   ```

2. **Close other GPU apps**: Stop browsers, games, video rendering

3. **For MPS**: Close memory-intensive apps (unified memory)

4. **Check available memory**:
   ```python
   import torch
   print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
   ```

### Power Monitoring Issues

**NVIDIA**: `pynvml not available`
```bash
pip install pynvml
# or
pip install nvidia-ml-py
```

**AMD**: `rocm-smi command not found`
```bash
# Ensure ROCm is installed
sudo apt install rocm-smi  # Ubuntu/Debian
```

**Apple**: `powermetrics permission denied`
```bash
# Requires sudo
sudo python benchmark.py --power
```

### Slow Performance

**Possible causes**:
1. **Thermal throttling**: Check temperatures
2. **Power mode**: Switch to "High Performance"
3. **Background apps**: Close GPU-intensive programs
4. **Driver issues**: Update GPU drivers
5. **PyTorch version**: Use latest stable (2.0+)

**Apple Silicon specific**:
1. **Battery mode**: Plug in Mac (40-60% performance drop on battery)
2. **Low Power Mode**: Disable in System Preferences
3. **Memory pressure**: Free up RAM (unified memory)

### Incorrect Results

**NVIDIA TF32 confusion**:
```python
# Ensure you're testing what you think
import torch
print(torch.get_float32_matmul_precision())
# Should be 'highest' for true FP32

# Force true FP32
torch.set_float32_matmul_precision('highest')
```

**Inconsistent timing**:
- Run multiple times (first run often slower)
- Check for background GPU usage
- Ensure proper synchronization

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

### Ways to Contribute

1. **Bug Reports**: Open issues with details
2. **Feature Requests**: Suggest improvements
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve guides and examples
5. **Benchmarks**: Share results for new GPUs

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Gpu-Ai-Bench.git
cd Gpu-Ai-Bench

# Create branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Make changes and test
pytest tests/

# Format code
black gpu_benchmark/
flake8 gpu_benchmark/

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### Code Style

- **PEP 8** compliance
- **Type hints** where possible
- **Docstrings** for all functions/classes
- **Comments** for complex logic
- **Tests** for new features

### Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**In short**: You can use, modify, and distribute this software freely, even for commercial purposes, as long as you include the original license.

---

## 📚 Citation

If you use this benchmark in your research or projects, please cite:

```bibtex
@software{gpu_ai_benchmark,
  author = {GPU AI Bench Contributors},
  title = {GPU AI Benchmark: Production-Grade GPU Benchmarking for AI Workloads},
  year = {2025},
  url = {https://github.com/Hiphex/Gpu-Ai-Bench},
  version = {1.0.0}
}
```

**Plain text**:
```
GPU AI Benchmark (2025)
https://github.com/Hiphex/Gpu-Ai-Bench
```

---

## 🙏 Acknowledgments

- **PyTorch Team**: For excellent cross-platform GPU support
- **NVIDIA**: For CUDA, cuDNN, and cuBLAS libraries
- **AMD**: For ROCm and open-source GPU computing
- **Apple**: For Metal Performance Shaders (MPS)
- **Community**: For feedback, bug reports, and contributions

### Inspired By

- MLPerf benchmarking suite
- Roofline Performance Model (Williams et al., 2009)
- Industry-standard GPU benchmarking practices
- Real-world AI workload optimization research

---

## 📞 Support

### Documentation
- **README**: This file (comprehensive guide)
- **API Reference**: Inline docstrings
- **Examples**: `examples/` directory (coming soon)

### Community
- **Issues**: [GitHub Issues](https://github.com/Hiphex/Gpu-Ai-Bench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Hiphex/Gpu-Ai-Bench/discussions)

### FAQ

**Q: Why is my performance lower than expected?**
A: Check thermal throttling, power mode, background apps, and driver versions.

**Q: Does this work on older GPUs?**
A: Yes! Requires compute capability 6.0+ (NVIDIA), GCN 3+ (AMD), or any Apple Silicon.

**Q: Can I benchmark multiple GPUs?**
A: Currently single-GPU. Multi-GPU support planned for v2.0.

**Q: How accurate are the results?**
A: Very accurate with GPU event timing and statistical analysis. Results within 1-2% variance.

**Q: What about INT8 quantization?**
A: INT8 testing available via FP8 benchmark (auto-converts). Native INT8 coming soon.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for the AI/ML community

[Report Bug](https://github.com/Hiphex/Gpu-Ai-Bench/issues) •
[Request Feature](https://github.com/Hiphex/Gpu-Ai-Bench/issues) •
[Contribute](CONTRIBUTING.md)

</div>
