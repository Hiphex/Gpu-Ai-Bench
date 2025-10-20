# Contributing to GPU AI Benchmark

Thank you for your interest in contributing to GPU AI Benchmark! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your system information (GPU model, driver version, PyTorch version)
- Error messages or logs

### Suggesting Features

Feature requests are welcome! Please open an issue with:
- A clear description of the feature
- Use case and benefits
- Any relevant examples or references

### Submitting Code

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test your changes**: Ensure benchmarks run without errors
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

## Testing

Before submitting:
- Run the benchmark on at least one GPU
- Test both NVIDIA and AMD paths if possible
- Ensure exports (JSON/CSV) work correctly
- Verify no memory leaks or crashes

## Adding New Benchmarks

When adding a new benchmark:

1. Create a new file in `gpu_benchmark/benchmarks/`
2. Follow the existing benchmark structure
3. Include warmup and test iterations
4. Add proper error handling
5. Update the main CLI to include your benchmark
6. Document the benchmark in README.md

Example structure:
```python
class NewBenchmark:
    def __init__(self, device: torch.device, warmup_iterations: int, test_iterations: int):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

    def run_all(self) -> Dict:
        # Implementation
        pass

    def get_results(self) -> Dict:
        return self.results

    def print_summary(self):
        # Print summary
        pass
```

## Questions?

Feel free to open an issue for any questions or clarifications!
