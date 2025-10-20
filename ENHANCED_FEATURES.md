# Enhanced Features - GPU AI Benchmark v2.0

This document covers the enhanced features added in v2.0, including real-world model benchmarks, thermal monitoring, baseline comparisons, and advanced analysis.

## Table of Contents

1. [Real-World Model Benchmarks](#real-world-model-benchmarks)
2. [Thermal Monitoring](#thermal-monitoring)
3. [Baseline Comparisons](#baseline-comparisons)
4. [Visualization & Reporting](#visualization--reporting)
5. [Usage Examples](#usage-examples)
6. [Performance Baselines Database](#performance-baselines-database)

---

## Real-World Model Benchmarks

The enhanced version includes industry-standard model benchmarks based on MLPerf specifications.

### ResNet50 Benchmark

Computer vision workload using ResNet50 architecture (~25M parameters).

**Metrics:**
- **Inference Throughput**: Images/second
- **Training Throughput**: Images/second
- **Latency**: Milliseconds per batch
- **AMP Speedup**: FP16 vs FP32 comparison (CUDA only)

**Configuration:**
```python
from gpu_benchmark.benchmarks.resnet_benchmark import ResNet50Benchmark

benchmark = ResNet50Benchmark(device=device, batch_size=32)
results = benchmark.run_all()
```

**Command Line:**
```bash
# Run ResNet50 benchmark with default batch size (32)
python benchmark_enhanced.py --resnet

# Custom batch size
python benchmark_enhanced.py --resnet --batch-size 64
```

**Example Output:**
```
ResNet50 Benchmark - Real-world Computer Vision Workload
============================================================

[1/3] Running inference benchmark (batch_size=32)...
  → Throughput: 245.67 images/sec
  → Latency: 130.25 ms

[2/3] Running training benchmark FP32 (batch_size=32)...
  → Throughput: 182.34 images/sec

[3/3] Running training benchmark AMP (batch_size=32)...
  → Throughput: 328.91 images/sec
  → AMP Speedup: 1.80x over FP32
```

### BERT Benchmark

NLP workload using BERT-Base architecture (~110M parameters).

**Metrics:**
- **Inference Throughput**: Sequences/second
- **Training Throughput**: Sequences/second
- **TTFT (Time to First Token)**: Milliseconds (approximation)
- **Latency**: Milliseconds per batch
- **AMP Speedup**: FP16 vs FP32 comparison

**Configuration:**
```python
from gpu_benchmark.benchmarks.bert_benchmark import BERTBenchmark

benchmark = BERTBenchmark(
    device=device,
    batch_size=8,
    seq_length=512
)
results = benchmark.run_all()
```

**Command Line:**
```bash
# Run BERT benchmark with defaults
python benchmark_enhanced.py --bert

# Custom configuration
python benchmark_enhanced.py --bert --bert-batch-size 16 --seq-length 256
```

**Example Output:**
```
BERT Benchmark - Real-world NLP Workload
Model: BERT-Base (12 layers, 768 hidden, ~110M params)
============================================================

[1/3] Running inference benchmark (batch_size=8, seq_len=512)...
  → Throughput: 24.56 sequences/sec
  → Latency: 325.67 ms
  → TTFT (approx): 27.14 ms

[2/3] Running training benchmark FP32...
  → Throughput: 15.23 sequences/sec

[3/3] Running training benchmark AMP...
  → Throughput: 28.45 sequences/sec
  → AMP Speedup: 1.87x over FP32
```

### Expected Performance Ranges

**ResNet50 Training (images/sec):**
- M3 Pro: 150-200
- RTX 3090: 700-900
- RTX 4090: 1100-1300
- A100 80GB: 1400-1600
- H100 80GB: 2800-3200

**BERT Training (sequences/sec, batch=8, seq=512):**
- M3 Pro: 20-30
- RTX 3090: 90-120
- RTX 4090: 160-200
- A100 80GB: 220-270
- H100 80GB: 450-550

---

## Thermal Monitoring

Real-time GPU temperature and throttling detection.

### Features

1. **Temperature Monitoring**: Real-time GPU temperature in Celsius
2. **Clock Speed Tracking**: Graphics and memory clock frequencies
3. **Throttle Detection**: Identifies thermal and power throttling
4. **Thermal Headroom**: Shows distance from throttle threshold

### Usage

```python
from gpu_benchmark.utils.thermal import ThermalMonitor

monitor = ThermalMonitor(device)

# Get current temperature
temp = monitor.get_temperature()  # Returns °C

# Get clock speeds
clocks = monitor.get_clock_speeds()
# {'graphics_clock_mhz': 2520, 'memory_clock_mhz': 9501}

# Check for throttling (NVIDIA only)
throttle = monitor.get_throttle_reasons()
# {
#   'thermal_limit': False,
#   'power_limit': False,
#   'hw_thermal_limit': False,
#   ...
# }

# Get thermal headroom
headroom = monitor.get_thermal_headroom()
# {
#   'current_temp_celsius': 72,
#   'slowdown_temp_celsius': 85,
#   'headroom_celsius': 13,
#   'headroom_percent': 15.3
# }

# Monitor during benchmark (10 seconds)
results = monitor.monitor_during_benchmark(duration_sec=10.0)
```

### Command Line

```bash
# Enable thermal monitoring
python benchmark_enhanced.py --all --thermal

# Output shows:
# Current Temperature: 68°C
# Slowdown Threshold: 85°C
# Thermal Headroom: 17°C (20.0%)
# Throttling Status: None
```

### Platform Support

| Platform | Temperature | Clock Speed | Throttle Detection |
|----------|-------------|-------------|-------------------|
| NVIDIA CUDA | ✅ | ✅ | ✅ |
| AMD ROCm | ⚠️ Limited | ⚠️ Limited | ❌ |
| Apple MPS | ❌ | ❌ | ❌ |

*Note: Apple Silicon thermal monitoring requires `sudo` access to `powermetrics`*

---

## Baseline Comparisons

Compare your GPU performance against 30+ known baselines.

### Features

1. **30+ GPU Baselines**: NVIDIA, AMD, and Apple Silicon
2. **Automatic Matching**: Find closest baseline by specs
3. **Multi-Metric Comparison**: FLOPS, bandwidth, real-world models
4. **Percentage Calculations**: See how your GPU stacks up

### Available Baselines

**NVIDIA:**
- RTX 40 Series: 4090, 4080, 4070 Ti
- RTX 30 Series: 3090, 3080, 3070
- Data Center: A100 40GB/80GB, H100 80GB

**AMD:**
- Consumer: RX 7900 XTX, RX 7900 XT
- Data Center: MI250X, MI300X

**Apple Silicon:**
- M1, M1 Pro, M1 Max
- M2, M2 Pro, M2 Max
- M3, M3 Pro, M3 Max
- M4, M4 Pro

### Usage

```python
from gpu_benchmark.utils.baselines import (
    get_baseline,
    find_closest_baseline,
    compare_to_baseline,
    list_baselines_by_vendor
)

# Get specific baseline
rtx_4090 = get_baseline("RTX 4090")
# {
#   'vendor': 'NVIDIA',
#   'fp32_tflops': 82.6,
#   'memory_gb': 24,
#   'resnet50_training_img_sec': 1200,
#   ...
# }

# Find closest match
closest = find_closest_baseline(
    fp32_tflops=7.4,
    memory_gb=36,
    vendor="Apple"
)
# Returns: ("M3 Pro", {...}, similarity_score)

# Compare measured results
comparison = compare_to_baseline(measured_results, "RTX 4090")
# {
#   'baseline_gpu': 'RTX 4090',
#   'comparisons': {
#     'fp32_tflops': {
#       'measured': 7.4,
#       'baseline': 82.6,
#       'percent_of_baseline': 8.96
#     },
#     ...
#   }
# }

# List available baselines
nvidia_gpus = list_baselines_by_vendor("NVIDIA")
# ['RTX 4090', 'RTX 4080', 'RTX 4070 Ti', ...]
```

### Command Line

```bash
# Compare against specific GPU
python benchmark_enhanced.py --all --compare "RTX 4090"

# Output:
# Baseline Comparison
# ======================================================================
# Comparing to: RTX 4090
#
#   fp32_tflops:
#     Measured: 7.42
#     Baseline: 82.60
#     Performance: 9.0% of baseline
#
#   memory_bandwidth_gbps:
#     Measured: 148.50
#     Baseline: 1008.00
#     Performance: 14.7% of baseline

# Auto-find closest baseline
python benchmark_enhanced.py --all --auto-compare

# Output:
# Auto-Detected Closest Baseline
# ======================================================================
# Closest match: M3 Pro (similarity score: 0.042)
#   Baseline FP32: 7.40 TFLOPS
#   Your FP32: 7.42 TFLOPS
#   Performance: 100.3% of baseline

# List all baselines
python benchmark_enhanced.py --list-baselines
```

---

## Visualization & Reporting

### ASCII Visualizations

Beautiful terminal-based charts and tables.

```python
from gpu_benchmark.utils.visualization import (
    create_ascii_bar_chart,
    create_comparison_table,
    create_performance_summary
)

# Bar chart
data = {
    "FP32": 10.5,
    "FP16": 21.0,
    "TF32": 15.5,
    "BF16": 20.5,
}
chart = create_ascii_bar_chart(data, "FLOPS Performance (TFLOPS)")

# Output:
# FLOPS Performance (TFLOPS)
# ==================================================================
# FP32 │ ████████████████████ 10.50
# FP16 │ ████████████████████████████████████████ 21.00
# TF32 │ ██████████████████████████████ 15.50
# BF16 │ ██████████████████████████████████████ 20.50
```

### HTML Reports

Export comprehensive HTML reports with tables and styling.

```bash
# Export HTML report
python benchmark_enhanced.py --all --html report.html
```

**Features:**
- Professional styling with CSS
- Organized sections (FLOPS, Memory, Workloads)
- Tables with hover effects
- Timestamp and GPU information
- Responsive design

### Performance Summary

```bash
# Show performance summary with ASCII visualizations
python benchmark_enhanced.py --all --summary
```

**Example Output:**
```
================================================================================
PERFORMANCE SUMMARY - M3 Pro
================================================================================

FLOPS Performance (TFLOPS)
==================================================================
FP32  │ ███████████████████ 7.42
FP16  │ ██████████████████████████████████████ 14.80

Memory Bandwidth (GB/s)
==================================================================
D2D   │ ████████████████████████████████████ 148.50
H2D   │ ███████████████████ 82.30
D2H   │ ██████████████████ 79.15

Real-World Workload Throughput
==================================================================
ResNet50 Inference    │ ████████████████████████ 245.67
ResNet50 Training     │ ██████████████████ 182.34
BERT Inference        │ ████████ 24.56
BERT Training         │ █████ 15.23

Thermal Information:
--------------------------------------------------------------------------------
  Average Temperature: 68.5 °C
  Max Temperature: 74.2 °C
  Throttling Detected: No

================================================================================
```

---

## Usage Examples

### Quick Start

```bash
# Install dependencies (including torchvision for ResNet)
pip install -r requirements.txt

# Run all enhanced benchmarks with summary
python benchmark_enhanced.py --all --summary

# Run specific benchmarks
python benchmark_enhanced.py --resnet --bert --summary
```

### Comprehensive Analysis

```bash
# Full analysis with thermal monitoring, baseline comparison, and HTML export
python benchmark_enhanced.py \
  --all \
  --thermal \
  --auto-compare \
  --summary \
  --html report.html \
  --json results.json
```

### Custom Configuration

```bash
# ResNet50 with larger batch size
python benchmark_enhanced.py \
  --resnet \
  --batch-size 64 \
  --iterations 200

# BERT with custom sequence length
python benchmark_enhanced.py \
  --bert \
  --bert-batch-size 16 \
  --seq-length 256

# Compare against multiple baselines
python benchmark_enhanced.py --all --compare "RTX 4090"
python benchmark_enhanced.py --all --compare "M3 Max"
python benchmark_enhanced.py --all --compare "MI300X"
```

### Programmatic Usage

```python
import torch
from gpu_benchmark.benchmarks.resnet_benchmark import ResNet50Benchmark
from gpu_benchmark.benchmarks.bert_benchmark import BERTBenchmark
from gpu_benchmark.utils.thermal import ThermalMonitor
from gpu_benchmark.utils.baselines import find_closest_baseline
from gpu_benchmark.utils.visualization import create_performance_summary

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")

# Run ResNet50 benchmark
resnet = ResNet50Benchmark(device, batch_size=32)
resnet_results = resnet.run_all()

# Run BERT benchmark
bert = BERTBenchmark(device, batch_size=8, seq_length=512)
bert_results = bert.run_all()

# Thermal monitoring
thermal = ThermalMonitor(device)
temp = thermal.get_temperature()
throttle = thermal.get_throttle_reasons()

# Find closest baseline
fp32_tflops = 7.4  # From FLOPS benchmark
memory_gb = 36
closest = find_closest_baseline(fp32_tflops, memory_gb)
print(f"Closest GPU: {closest[0]}")

# Create summary visualization
results = {
    "resnet50": resnet_results,
    "bert": bert_results,
}
summary = create_performance_summary(results, "M3 Pro")
print(summary)
```

---

## Performance Baselines Database

### Data Sources

Baseline data compiled from:
1. **MLPerf Results**: Official MLPerf training and inference benchmarks
2. **Lambda GPU Benchmarks**: PyTorch training throughput data
3. **Vendor Specifications**: Official TFLOPS and bandwidth specs
4. **Community Benchmarks**: Verified results from reputable sources

### Baseline Data Structure

Each baseline includes:

```python
{
    "vendor": "NVIDIA",                    # GPU vendor
    "architecture": "Ada Lovelace",        # GPU architecture
    "fp32_tflops": 82.6,                   # FP32 peak performance
    "fp16_tflops": 165.2,                  # FP16 peak performance
    "tensor_tflops": 660.6,                # Tensor core performance
    "memory_gb": 24,                       # Total memory
    "memory_bandwidth_gbps": 1008,         # Memory bandwidth
    "tdp_watts": 450,                      # Thermal design power
    "resnet50_training_img_sec": 1200,     # ResNet50 training throughput
    "bert_training_seq_sec": 180,          # BERT training throughput
}
```

### Top Performers by Category

**Highest FP32 TFLOPS:**
1. MI300X: 163 TFLOPS
2. RTX 4090: 82.6 TFLOPS
3. H100: 67 TFLOPS

**Highest Memory Bandwidth:**
1. MI300X: 5300 GB/s
2. H100: 3352 GB/s
3. MI250X: 3277 GB/s

**Best Performance per Watt (TFLOPS/W):**
1. M3 Pro: 0.21 TFLOPS/W
2. M4: 0.20 TFLOPS/W
3. M3 Max: 0.20 TFLOPS/W

**ResNet50 Training Leaders:**
1. MI300X: 3500 images/sec
2. H100: 3000 images/sec
3. A100 80GB: 1500 images/sec

---

## System Requirements

### Enhanced Features Dependencies

```bash
# Core requirements
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
psutil>=5.9.0

# Enhanced features
torchvision>=0.15.0  # For ResNet50 benchmark
pynvml>=11.5.0       # For NVIDIA thermal monitoring
matplotlib>=3.7.0    # Optional: for future plotting features
```

### Platform Support

| Feature | CUDA | ROCm | MPS |
|---------|------|------|-----|
| ResNet50 Benchmark | ✅ | ✅ | ✅ |
| BERT Benchmark | ✅ | ✅ | ✅ |
| AMP (Mixed Precision) | ✅ | ⚠️ | ❌ |
| Thermal Monitoring | ✅ | ⚠️ | ❌ |
| Baseline Comparison | ✅ | ✅ | ✅ |
| ASCII Visualization | ✅ | ✅ | ✅ |
| HTML Export | ✅ | ✅ | ✅ |

✅ Fully supported | ⚠️ Limited support | ❌ Not supported

---

## Performance Tips

### ResNet50 Optimization

1. **Batch Size**: Larger batches = better throughput (until memory limit)
2. **AMP**: Use `--all` to compare FP32 vs AMP (1.5-2x speedup on CUDA)
3. **Channels Last**: Consider channels-last memory format for convolutions

### BERT Optimization

1. **Sequence Length**: Shorter sequences = higher throughput
2. **Batch Size**: Balance between throughput and memory usage
3. **AMP**: Essential for competitive BERT performance (1.5-2x speedup)

### Thermal Management

1. **Monitor Temperature**: Keep below 80°C for sustained performance
2. **Check Throttling**: Thermal throttling can reduce performance by 20-40%
3. **Cooling**: Ensure adequate airflow/cooling during benchmarks
4. **Power Limits**: Increase power limit if thermal headroom available

---

## Troubleshooting

### ResNet50 Import Error

```
ImportError: torchvision not available
```

**Solution:**
```bash
pip install torchvision>=0.15.0
```

### Thermal Monitoring Not Working

```
Temperature: None
```

**NVIDIA**: Install pynvml
```bash
pip install pynvml
```

**AMD**: Ensure `rocm-smi` is in PATH

**Apple**: Thermal monitoring requires `sudo powermetrics` (not implemented in basic version)

### AMP Not Working on MPS

AMP (Automatic Mixed Precision) is only supported on CUDA. The benchmark will automatically skip AMP on MPS/CPU.

### Low Performance Compared to Baseline

Possible causes:
1. **Thermal Throttling**: Check with `--thermal`
2. **Power Limit**: GPU may be power-limited
3. **Battery Mode**: Mac laptops perform 40-60% worse on battery
4. **Background Processes**: Close other GPU applications
5. **Driver Version**: Ensure latest GPU drivers

---

## Future Enhancements

Planned for v2.1+:

- [ ] Real LLM benchmarks (GPT-2, Llama)
- [ ] Image generation benchmarks (Stable Diffusion)
- [ ] Multi-GPU scaling tests
- [ ] Interactive matplotlib charts
- [ ] Automatic performance regression detection
- [ ] Cloud GPU pricing integration (cost per FLOP)
- [ ] Benchmark result database (submit/compare with community)

---

## Citation

If you use this tool in your research or publications, please cite:

```bibtex
@software{gpu_ai_benchmark_enhanced,
  title = {GPU AI Benchmark - Enhanced Edition},
  author = {Hiphex},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/Hiphex/Gpu-Ai-Bench}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- **MLCommons** for MLPerf benchmark specifications
- **Lambda Labs** for GPU benchmark data
- **PyTorch Team** for excellent deep learning framework
- **NVIDIA, AMD, Apple** for GPU compute platforms
