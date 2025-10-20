# Advanced Features

This document describes the advanced features added to the GPU AI Benchmark tool.

## Table of Contents
1. [Precision Control](#precision-control)
2. [Enhanced Timing](#enhanced-timing)
3. [Memory Bandwidth Analysis](#memory-bandwidth-analysis)
4. [Multi-Vendor Power Monitoring](#multi-vendor-power-monitoring)
5. [Roofline Analysis](#roofline-analysis)
6. [SDPA Backend Testing](#sdpa-backend-testing)
7. [Statistics & Reporting](#statistics--reporting)

---

## 1. Precision Control

### TF32 vs FP32 on NVIDIA GPUs

NVIDIA Ampere and newer GPUs support TensorFloat-32 (TF32), which provides ~8x speedup over true FP32 with minimal accuracy loss. The benchmark now explicitly controls this:

```python
# True FP32 (highest precision)
torch.set_float32_matmul_precision("highest")

# TF32 (faster, still accurate for most AI workloads)
torch.set_float32_matmul_precision("high")
```

**Usage:**
```bash
# Benchmark with both FP32 and TF32
python benchmark.py --flops --tf32

# Only true FP32
python benchmark.py --flops --no-tf32
```

### Supported Precisions

| Precision | Description | Availability |
|-----------|-------------|--------------|
| **FP32** | 32-bit floating point (true) | All GPUs |
| **TF32** | TensorFloat-32 | NVIDIA Ampere+ |
| **BF16** | Brain Float 16 | Modern GPUs |
| **FP16** | 16-bit floating point | All GPUs |
| **FP8** | 8-bit floating point | NVIDIA Hopper+, some AMD |

---

## 2. Enhanced Timing

### GPU Event-Based Timing

For accurate GPU performance measurement, we use hardware events instead of CPU wall-clock time:

**CUDA/ROCm:**
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... operation ...
end.record()
torch.cuda.synchronize()

time_ms = start.elapsed_time(end)  # Nanosecond-precision GPU timer
```

**MPS (Apple Silicon):**
```python
# MPS doesn't support events, use wall-clock with synchronization
torch.mps.synchronize()
start = time.perf_counter()
# ... operation ...
torch.mps.synchronize()
elapsed = time.perf_counter() - start
```

### Statistical Analysis

Instead of just mean time, we report:
- **Median**: More robust to outliers
- **P10/P90**: 10th and 90th percentiles
- **Standard Deviation**: Variability measure
- **Min/Max**: Performance range

**Example output:**
```
Median: 25.43 TFLOPS (24.12 ms)
Mean:   25.01 TFLOPS (24.53 ms)
Std:    0.32 ms
Range:  [23.89 - 24.95] ms (p10-p90)
```

---

## 3. Memory Bandwidth Analysis

### Pinned vs Pageable Memory

**Pinned (Page-Locked) Memory** provides significantly faster Host↔Device transfers:

```python
# Pageable memory (default)
data = torch.randn(size)

# Pinned memory (CUDA only)
data = torch.randn(size, pin_memory=True)
```

**Performance Impact:**
- **Pinned**: Up to 2-3x faster H2D/D2H transfers
- **Pageable**: Slower but doesn't lock system RAM
- **MPS**: Unified memory architecture, pinning not applicable

**Usage:**
```bash
# Compare pinned vs pageable
python benchmark.py --memory --compare-pinning
```

### Bandwidth Types

1. **Device-to-Device (D2D)**: Internal GPU memory bandwidth
2. **Host-to-Device (H2D)**: CPU → GPU transfer speed
3. **Device-to-Host (D2H)**: GPU → CPU transfer speed

---

## 4. Multi-Vendor Power Monitoring

### NVIDIA (pynvml / nvidia-ml-py)

```bash
pip install pynvml
```

**Metrics:**
- Power draw (Watts)
- Power limit
- Temperature
- Fan speed
- Energy consumption (J = ∫W·dt)

### AMD (rocm-smi)

```bash
# Requires ROCm installed
rocm-smi --showpower
```

**Metrics:**
- Average Graphics Package Power
- GPU temperature
- Clock speeds

### Apple Silicon (powermetrics)

```bash
# Requires sudo on macOS
sudo powermetrics --samplers gpu_power -i 100 -n 1
```

**Metrics:**
- GPU Power (mW)
- ANE (Neural Engine) Power
- Integrated power reporting

**Note**: Power monitoring requires appropriate permissions:
- **NVIDIA**: Just install pynvml
- **AMD**: ROCm must be installed
- **Apple**: Requires sudo access

---

## 5. Roofline Analysis

The Roofline Model helps understand if operations are **compute-bound** or **memory-bound**.

### Concepts

**Operational Intensity** = FLOPs / Bytes Accessed

**Two Performance Limits:**
1. **Peak FLOP/s** (horizontal roof) - Compute limit
2. **Peak Bandwidth × Intensity** (sloped roof) - Memory limit

**Ridge Point** = Peak_FLOPS / Peak_Bandwidth

- **Below ridge**: Memory-bound (improve data reuse)
- **Above ridge**: Compute-bound (improve algorithm)

### Example Analysis

```
Matrix Multiplication (4096×4096):
  Operational Intensity: 341.3 FLOP/byte
  Ridge Point: 41.0 FLOP/byte
  → Compute-bound (above ridge)
  → Efficiency: 87% of theoretical peak
```

### Usage

```python
from gpu_benchmark.utils.roofline import RooflineAnalysis

# Create analyzer with your GPU specs
roofline = RooflineAnalysis(
    peak_flops=82580,      # GFLOP/s (e.g., RTX 4090 FP32)
    peak_bandwidth_gbs=1008  # GB/s
)

# Analyze a kernel
analysis = roofline.analyze_matmul(
    M=4096, N=4096, K=4096,
    time_s=0.025,
    dtype_bytes=4  # FP32
)

print(f"Bound by: {analysis['bound_by']}")
print(f"Efficiency: {analysis['efficiency_percent']:.1f}%")
```

---

## 6. SDPA Backend Testing

PyTorch 2.0+ supports multiple backends for Scaled Dot Product Attention:

### Available Backends

1. **FlashAttention**: Fastest, requires compatible GPU (Ampere+)
2. **Memory-Efficient**: Slower than Flash but more compatible
3. **Math (Baseline)**: Pure PyTorch implementation, slowest

### Benchmark Output

```
BERT-like (batch=16, seq=512):
  FlashAttention............... 12.34 ms (661,504 tokens/s)
  Memory-Efficient............. 18.76 ms (435,200 tokens/s)
  Math (Baseline).............. 45.23 ms (180,480 tokens/s)
```

### Usage

```bash
python benchmark.py --sdpa
```

**Automatic Detection**: The benchmark automatically detects which backends are available on your hardware and only tests those.

---

## 7. Statistics & Reporting

### Enhanced Metrics

Every benchmark now reports:

| Metric | Description | Why Important |
|--------|-------------|---------------|
| Median | Middle value when sorted | Robust to outliers |
| Mean | Average | Traditional metric |
| Std Dev | Variability | Consistency indicator |
| P10 | 10th percentile | Fast cases |
| P90 | 90th percentile | Slow cases |
| Min/Max | Absolute range | Extreme performance |

### JSON Export

Full results with all metadata:

```json
{
  "timestamp": "2025-10-20T18:30:00",
  "gpu_info": {...},
  "precision_settings": {
    "float32_matmul_precision": "highest",
    "cudnn_allow_tf32": false
  },
  "flops": {
    "fp32": {
      "median_tflops": 82.5,
      "mean_tflops": 82.1,
      "detailed_results": [...]
    }
  },
  "roofline": {...},
  "power": {
    "total_energy_j": 5432.1,
    "avg_power_w": 425.6
  }
}
```

### CSV Export

Simplified format for spreadsheet analysis and comparisons across multiple runs.

---

## CLI Usage Examples

```bash
# Full enhanced benchmark
python benchmark.py --all --tf32 --compare-pinning --roofline

# Precision comparison
python benchmark.py --flops --tf32 --precision fp32,tf32,bf16,fp16

# Memory analysis
python benchmark.py --memory --compare-pinning --sizes 128,512,2048

# Power efficiency
python benchmark.py --power --duration 30  # 30-second stress test

# Export everything
python benchmark.py --all --json results.json --csv results.csv
```

---

## Performance Tips by Feature

### For Best FLOPS Numbers
1. Use TF32 on NVIDIA Ampere+ for practical performance
2. Use FP16 for maximum throughput
3. Ensure thermal headroom (good cooling)
4. Close background applications

### For Best Memory Bandwidth
1. Use pinned memory for H2D/D2H (CUDA)
2. Use larger transfer sizes
3. On MPS, free up unified memory

### For Accurate Power Measurements
1. Let GPU warm up (first runs may be anomalous)
2. Run longer tests (30+ seconds) for stable averages
3. Ensure proper permissions (sudo for Apple)

### For SDPA Performance
1. FlashAttention requires Ampere or newer
2. Sequence length affects backend selection
3. FP16/BF16 is much faster than FP32

---

## Developer API

All utilities are importable:

```python
from gpu_benchmark.utils import (
    DeviceTimer,
    benchmark_operation,
    MatmulPrecisionContext,
    RooflineAnalysis,
    create_power_monitor
)

# Use in your own benchmarks
timer = DeviceTimer(device)
with timer:
    result = my_operation()
elapsed_ms = timer.elapsed_time_ms()
```

---

## Future Enhancements

Planned features:
- [ ] Multi-GPU scaling tests (NCCL/RCCL)
- [ ] Real model benchmarks (ResNet50, small LLMs)
- [ ] MLPerf integration hooks
- [ ] Interactive roofline plots
- [ ] KV-cache decode microbenchmarks
- [ ] Triton kernel examples

Contributions welcome!
