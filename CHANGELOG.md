# Changelog

All notable changes to GPU AI Benchmark will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-20

### Added

#### Core Features
- **Multi-Platform GPU Support**: NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS)
- **Comprehensive FLOPS Benchmarking**: FP32, FP16, FP8/INT8
- **Memory Bandwidth Testing**: D2D, H2D, D2H transfers
- **Power Monitoring**: NVIDIA GPUs via pynvml
- **AI Workload Benchmarks**: Convolutions, attention, element-wise, reductions
- **JSON/CSV Export**: Full results with metadata

#### Production-Grade Enhancements (v1.0.0)
- **Advanced Timing System**:
  - GPU event-based timing for CUDA/ROCm (nanosecond precision)
  - Wall-clock + synchronization for MPS
  - Statistical analysis: median, p10, p90, std dev, min, max

- **Precision Control**:
  - Explicit TF32 vs FP32 control on NVIDIA GPUs
  - BF16 (Brain Float 16) support
  - FP8 detection and testing (Hopper+)
  - MatmulPrecisionContext API

- **Enhanced Memory Analysis**:
  - Pinned vs pageable memory comparison
  - 2-3x speedup demonstration on CUDA
  - Unified memory support for Apple Silicon
  - Memory transfer utilities module

- **Multi-Vendor Power Monitoring**:
  - NVIDIA: pynvml (refactored and enhanced)
  - AMD: rocm-smi CLI integration
  - Apple: powermetrics support (requires sudo)
  - Energy integration (Joules = ∫Watts·dt)
  - Performance per Watt metrics

- **Roofline Analysis**:
  - Compute vs memory bound detection
  - Operational intensity calculation (FLOP/byte)
  - Ridge point identification
  - Efficiency percentage reporting
  - GPU specs database for major GPUs

- **SDPA Backend Testing**:
  - FlashAttention support (Ampere+)
  - Memory-Efficient Attention
  - Math baseline (PyTorch)
  - Automatic backend detection
  - Tokens/second throughput

#### Apple Silicon Support
- **MPS Backend Integration**: Full support for M1/M2/M3/M4 chips
- **Unified Memory Handling**: Proper detection and reporting
- **System Info Extraction**: CPU brand, memory, GPU family via sysctl/system_profiler
- **Performance Optimization Tips**: Battery, thermal, memory pressure guidance

#### Developer API
- **DeviceTimer**: Cross-platform GPU timer class
- **benchmark_operation**: Standard benchmarking function with stats
- **MatmulPrecisionContext**: Context manager for precision control
- **RooflineAnalysis**: Roofline model implementation
- **PowerMonitor**: Abstract base class with vendor-specific implementations
- **Memory utilities**: Pinned vs pageable transfer functions

#### Documentation
- **Comprehensive README**: 1000+ lines with full guide
- **Advanced Features Guide**: Detailed explanations and examples
- **API Documentation**: Inline docstrings for all modules
- **Performance Tips**: Platform-specific optimization guidance
- **Troubleshooting Section**: Common issues and solutions
- **Citation Guide**: BibTeX and plain text formats

### Fixed
- **Apple Silicon Issues**:
  - Memory detection showing 0.00 GB → Now shows actual unified memory
  - Host-to-device transfer errors → Disabled pinned memory on MPS
  - GPU name showing N/A → Added proper name field
  - Banner not mentioning MPS → Updated to include Apple Silicon

- **Timing Accuracy**:
  - CPU-GPU synchronization overhead → GPU events eliminate this
  - Inconsistent results → Statistical analysis reveals variance
  - First-run slowness → Explicit warmup iterations

- **Memory Bandwidth**:
  - Incorrect H2D/D2H speeds → Fixed with proper pinned memory handling
  - MPS "missing kernel" errors → Graceful degradation for unsupported features

### Changed
- **Benchmark Output**: Now shows median + percentiles instead of just mean
- **Result Format**: Enhanced JSON with precision settings and detailed stats
- **Error Messages**: More helpful with platform-specific guidance

### Performance
- **Apple M3 Pro Results** (validated):
  - FP32: ~4 TFLOPS
  - FP16: ~4.3 TFLOPS
  - Memory: ~57 GB/s unified
  - Convolution: ~4,745 img/s
  - Attention: ~527,254 tokens/s

### Technical Debt
- Separated v2 benchmarks from v1 (gradual integration planned)
- ADVANCED_FEATURES.md merged into README (single source of truth)
- Utils module created for shared functionality

## [Future] - Planned

### v1.1.0 (Next Release)
- [ ] Integrate enhanced benchmarks into main CLI
- [ ] Add --tf32, --roofline, --compare-pinning flags
- [ ] Multi-GPU scaling tests (NCCL/RCCL)
- [ ] Interactive roofline plots (matplotlib)
- [ ] Examples directory with common use cases

### v1.2.0
- [ ] Real model benchmarks (ResNet50, BERT, small LLMs)
- [ ] MLPerf integration hooks
- [ ] KV-cache decode microbenchmarks
- [ ] Triton kernel examples

### v2.0.0 (Major)
- [ ] Multi-GPU support (data parallel, model parallel)
- [ ] Distributed benchmarking
- [ ] Web UI for results visualization
- [ ] Database storage for historical comparisons
- [ ] CI/CD integration examples

---

## Version History

**v1.0.0** (2025-10-20): Production-grade release with advanced features
- Initial public release
- Full cross-platform support
- Comprehensive documentation

---

## Links

- [GitHub Repository](https://github.com/Hiphex/Gpu-Ai-Bench)
- [Issue Tracker](https://github.com/Hiphex/Gpu-Ai-Bench/issues)
- [Contributing Guide](CONTRIBUTING.md)
