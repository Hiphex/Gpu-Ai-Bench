# GPU AI Benchmark

A comprehensive, production-grade GPU benchmarking tool designed specifically for AI workloads. Measures TFLOPS performance (FP32/TF32/BF16/FP16/FP8), memory bandwidth, power consumption, and real-world AI operation throughput.

**Supports**: NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (M1/M2/M3/M4)

## ✨ Key Features

- **Multi-Platform Support**: Works with NVIDIA CUDA, AMD ROCm, and Apple Silicon (MPS) GPUs
- **Precision Testing**: FP32, TF32 (NVIDIA), BF16, FP16, and FP8 operations
- **Advanced Timing**: GPU event-based timing (CUDA/ROCm) with statistical analysis (median, p10/p90)
- **Memory Analysis**: Pinned vs pageable transfers, bandwidth measurement (D2D, H2D, D2H)
- **Power Monitoring**: Multi-vendor support (NVIDIA pynvml, AMD rocm-smi, Apple powermetrics)
- **Roofline Analysis**: Understand compute vs memory bottlenecks
- **SDPA Backends**: Test FlashAttention, Memory-Efficient, and Math backends
- **Export Results**: JSON and CSV with comprehensive metadata

📖 **[See Advanced Features Documentation](ADVANCED_FEATURES.md)** for detailed information about precision control, roofline analysis, power monitoring, and more.

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+ with GPU support (CUDA, ROCm, or MPS)
- **NVIDIA GPUs**: CUDA drivers
- **AMD GPUs**: ROCm support
- **Apple Silicon**: M1/M2/M3/M4 with macOS 12.3+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hiphex/Gpu-Ai-Bench.git
cd Gpu-Ai-Bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For NVIDIA GPU power monitoring, also install:
```bash
pip install pynvml
```

3. Ensure PyTorch is installed with GPU support:
```bash
# For NVIDIA CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For AMD ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# For Apple Silicon (M1/M2/M3/M4) - use default PyTorch with MPS support
pip install torch torchvision torchaudio
```

**Verify GPU is detected:**
```bash
# For NVIDIA/AMD
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For Apple Silicon
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Usage

### Run All Benchmarks

```bash
python benchmark.py --all
```

Or using the module directly:

```bash
python -m gpu_benchmark.main --all
```

### Run Specific Benchmarks

```bash
# FLOPS only
python benchmark.py --flops

# Memory bandwidth only
python benchmark.py --memory

# Power consumption only
python benchmark.py --power

# AI workloads only
python benchmark.py --workload

# Combine multiple benchmarks
python benchmark.py --flops --memory --workload
```

### Export Results

Export to JSON:
```bash
python benchmark.py --all --json
```

Export to CSV:
```bash
python benchmark.py --all --csv
```

Export to both with custom filenames:
```bash
python benchmark.py --all --json my_results.json --csv my_results.csv
```

Specify output directory:
```bash
python benchmark.py --all --json --csv --output-dir ./my_results
```

### Advanced Options

Customize iteration counts:
```bash
python benchmark.py --all --warmup 20 --iterations 200
```

## Benchmarks Explained

### 1. FLOPS Benchmark
Measures theoretical peak performance in TFLOPS (Trillions of Floating Point Operations Per Second) for different precision formats:
- **FP32**: Standard single precision
- **FP16**: Half precision (common in AI training)
- **FP8**: 8-bit precision (newer GPUs, maximum AI throughput)

### 2. Memory Benchmark
Tests memory subsystem performance:
- **Total Memory**: Available GPU memory in GB
- **Device-to-Device Bandwidth**: Internal GPU memory transfers
- **Host-to-Device Bandwidth**: CPU to GPU transfer speed
- **Device-to-Host Bandwidth**: GPU to CPU transfer speed

### 3. Power Benchmark
Monitors power consumption (**NVIDIA GPUs only** - requires `pynvml`):
- **Power Limit**: Maximum allowed power draw
- **Idle Power**: Power consumption at rest
- **Stress Test**: Power draw under full load
- **Temperature Monitoring**: GPU temperature during tests

**Note**: Power monitoring is not available on AMD or Apple Silicon GPUs.

### 4. AI Workload Benchmark
Tests real-world AI operations:
- **Convolution**: CNN operations (image processing, computer vision)
- **Attention**: Transformer operations (NLP, large language models)
- **Element-wise Operations**: Activations (ReLU, GELU, Sigmoid, Tanh)
- **Reduction Operations**: Sum, mean, max operations

## Understanding the Results

### JSON Output Structure
```json
{
  "timestamp": "2025-01-20T10:30:00",
  "gpu_info": {
    "name": "NVIDIA RTX 4090",
    "total_memory_gb": 24.0,
    "compute_capability": "8.9"
  },
  "flops": {
    "fp16": {"average_tflops": 165.43},
    "fp32": {"average_tflops": 82.58}
  },
  "memory": {
    "bandwidth": {"peak_bandwidth_gbs": 950.23}
  },
  "power": {
    "stress_test": {"avg_power_w": 425.5}
  },
  "workloads": {
    "convolution": {"average_throughput_img_per_sec": 1250.5}
  }
}
```

### CSV Output
The CSV file contains a flattened view with one row per metric, making it easy to compare results across multiple runs or GPUs.

## Performance Tips

### General
1. **Close Background Applications**: For accurate results, close other GPU-intensive applications
2. **Thermal Management**: Ensure adequate cooling for sustained performance
3. **Power Settings**: Use maximum performance power mode
4. **Multiple Runs**: Run the benchmark 2-3 times and average the results for consistency

### Apple Silicon Specific
1. **Plug In Your Mac**: Performance is significantly reduced on battery power
2. **Free Up RAM**: Unified memory is shared - close memory-intensive apps (Chrome, Docker, etc.)
3. **Disable Low Power Mode**: System Preferences > Battery > disable Low Power Mode
4. **Allow Thermal Stabilization**: First run may be slower, subsequent runs are faster
5. **Expected Performance Ranges**:
   - **M1/M2**: 3-6 TFLOPS FP32, 10-20 TFLOPS FP16
   - **M1/M2 Pro/Max**: 5-10 TFLOPS FP32, 15-30 TFLOPS FP16
   - **M3/M3 Pro/Max**: 6-12 TFLOPS FP32, 20-40 TFLOPS FP16
   - **M4**: 8-15 TFLOPS FP32, 25-50 TFLOPS FP16

**Note**: MPS backend may show lower performance than theoretical peak due to optimization maturity. CUDA has decades of optimization while MPS is relatively new.

## Troubleshooting

### No GPU Detected
```
Error: No compatible GPU detected
```
**Solution**: Ensure PyTorch is installed with GPU support. Check installation:
```bash
# For NVIDIA/AMD
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# For Apple Silicon
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

**Apple Silicon Note**: If MPS is not available, ensure you have:
- macOS 12.3 or later
- PyTorch 2.0 or later
- Run: `pip install --upgrade torch torchvision torchaudio`

### Out of Memory Errors
```
RuntimeError: CUDA/MPS out of memory
```
**Solution**: Close other GPU applications or reduce test iterations:
```bash
python benchmark.py --all --iterations 50
```

**Apple Silicon Note**: Unified memory is shared between CPU and GPU. Close memory-intensive applications to free up more RAM for GPU use.

### Power Monitoring Not Available
```
Warning: pynvml not available
```
**Solution**: Install NVIDIA Management Library:
```bash
pip install pynvml
```

## Comparing GPUs

To compare different GPUs:

1. Run benchmark on each GPU
2. Export results with descriptive names:
```bash
# On system with RTX 4090
python benchmark.py --all --json rtx4090_results.json

# On system with RTX 3090
python benchmark.py --all --json rtx3090_results.json
```

3. Compare the JSON files to see performance differences

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with PyTorch for cross-platform GPU support
- Inspired by industry-standard GPU benchmarking practices
- Based on research into AI workload performance metrics

## Citation

If you use this benchmark in your research or projects, please cite:

```
GPU AI Benchmark (2025)
https://github.com/Hiphex/Gpu-Ai-Bench
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the documentation
- Review closed issues for solutions

---

**Note**: Benchmark results can vary based on system configuration, driver versions, cooling, and power settings. Always run multiple iterations for consistent results.
