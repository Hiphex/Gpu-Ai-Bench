# GPU AI Benchmark

A comprehensive GPU benchmarking tool designed specifically for AI workloads. Measures TFLOPS performance (FP16/FP8), memory bandwidth, power consumption, and real-world AI operation throughput.

## Features

- **Multi-Platform Support**: Works with both NVIDIA CUDA and AMD ROCm GPUs
- **Precision Testing**: Benchmarks FP32, FP16, and FP8 operations
- **Memory Analysis**: Measures bandwidth (device-to-device, host-to-device, device-to-host)
- **Power Monitoring**: Tracks power consumption during idle and stress tests (NVIDIA GPUs)
- **Real-World AI Workloads**: Tests convolutions, attention mechanisms, and common operations
- **Export Results**: Save results to JSON and CSV formats
- **Detailed Reporting**: Comprehensive performance metrics and summaries

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+ with CUDA or ROCm support
- NVIDIA GPU with CUDA drivers (for NVIDIA GPUs)
- AMD GPU with ROCm support (for AMD GPUs)

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
Monitors power consumption (NVIDIA GPUs only):
- **Power Limit**: Maximum allowed power draw
- **Idle Power**: Power consumption at rest
- **Stress Test**: Power draw under full load
- **Temperature Monitoring**: GPU temperature during tests

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

1. **Close Background Applications**: For accurate results, close other GPU-intensive applications
2. **Thermal Management**: Ensure adequate cooling for sustained performance
3. **Power Settings**: Use maximum performance power mode
4. **Multiple Runs**: Run the benchmark 2-3 times and average the results for consistency

## Troubleshooting

### No GPU Detected
```
Error: No CUDA-compatible GPU detected
```
**Solution**: Ensure PyTorch is installed with CUDA/ROCm support. Check installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Errors
```
RuntimeError: CUDA out of memory
```
**Solution**: Close other GPU applications or reduce test iterations:
```bash
python benchmark.py --all --iterations 50
```

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
