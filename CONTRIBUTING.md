# Contributing to GPU AI Benchmark

Thank you for your interest in contributing to GPU AI Benchmark! We welcome contributions from the community to make this tool even better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Adding New Features](#adding-new-features)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together and help each other
- **Be inclusive**: Welcome contributors of all backgrounds and skill levels
- **Be professional**: Focus on the technical merits of contributions

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug? Please open an issue with:

**Required Information:**
- **Description**: Clear description of the problem
- **Steps to Reproduce**: Detailed steps to reproduce the bug
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **System Information**:
  ```bash
  - GPU Model: (e.g., NVIDIA RTX 4090, Apple M3 Pro)
  - OS: (e.g., Ubuntu 22.04, macOS 14.0)
  - Python Version: (e.g., 3.10.5)
  - PyTorch Version: (e.g., 2.0.1)
  - CUDA/ROCm Version: (if applicable)
  ```
- **Error Messages**: Full error traceback
- **Screenshots**: If applicable

**Example:**
```markdown
## Bug Report

**Description**: Memory benchmark crashes on Apple Silicon

**Steps to Reproduce**:
1. Run `python benchmark.py --memory`
2. Observe crash during H2D transfer

**Expected**: Should complete successfully
**Actual**: RuntimeError: missing kernel for mps

**System**: Apple M3 Pro, macOS 14.0, PyTorch 2.1.0
```

### 💡 Suggesting Features

Have an idea? We'd love to hear it! Please open an issue with:

- **Feature Description**: What you want to add
- **Use Case**: Why this feature is valuable
- **Implementation Ideas**: How it might work (optional)
- **Examples**: Similar features in other tools (optional)
- **Alternatives**: Other approaches you considered

### 🔧 Code Contributions

We accept pull requests for:
- Bug fixes
- New benchmarks
- Performance improvements
- Documentation improvements
- New platform support
- Utility functions
- Test coverage

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- GPU with drivers (NVIDIA/AMD/Apple)
- PyTorch 2.0+ with GPU support

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Gpu-Ai-Bench.git
cd Gpu-Ai-Bench

# Add upstream remote
git remote add upstream https://github.com/Hiphex/Gpu-Ai-Bench.git
```

### Create Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-short-description
```

### Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Install PyTorch with GPU support
# (See README for platform-specific instructions)
```

### Verify Setup

```bash
# Run existing benchmarks
python benchmark.py --flops

# Check GPU detection
python -c "import torch; print(f'GPU: {torch.cuda.is_available() or (hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available())}')"
```

---

## Coding Guidelines

### Code Style

We follow **PEP 8** with some project-specific conventions:

#### Formatting

```bash
# Format code with Black (line length: 100)
black --line-length 100 gpu_benchmark/

# Check style with flake8
flake8 gpu_benchmark/ --max-line-length=100
```

#### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, List, Optional

def benchmark_operation(
    operation: Callable,
    warmup_iterations: int = 10,
    test_iterations: int = 100,
    device: torch.device = None,
    return_stats: bool = True
) -> Dict[str, float]:
    """
    Benchmark an operation with proper warmup and statistics

    Args:
        operation: Callable that performs the operation
        warmup_iterations: Number of warmup iterations
        test_iterations: Number of test iterations
        device: Device to use for timing
        return_stats: Whether to return detailed statistics

    Returns:
        Dictionary containing timing statistics
    """
    # Implementation
    pass
```

#### Docstrings

Use **Google-style** docstrings:

```python
def analyze_matmul(self, M: int, N: int, K: int, time_s: float) -> Dict:
    """
    Analyze matrix multiplication performance

    Args:
        M: Rows in first matrix
        N: Columns in second matrix
        K: Columns in first matrix / rows in second
        time_s: Time taken in seconds

    Returns:
        Dict with roofline analysis including:
        - operational_intensity: FLOPs per byte
        - efficiency_percent: Percentage of theoretical peak
        - bound_by: 'compute' or 'memory'

    Raises:
        ValueError: If dimensions are invalid
    """
    pass
```

### Project Structure

Follow the existing structure:

```
gpu_benchmark/
├── benchmarks/       # Benchmark implementations
│   ├── flops_benchmark.py
│   ├── memory_benchmark.py
│   └── your_new_benchmark.py
├── utils/            # Utility modules
│   ├── __init__.py
│   ├── timing.py
│   ├── power.py
│   └── roofline.py
├── gpu_detector.py   # GPU detection
├── exporter.py       # Results export
└── main.py           # CLI interface
```

### Error Handling

Be defensive and handle errors gracefully:

```python
def benchmark_with_precision(dtype: torch.dtype) -> Dict:
    try:
        # Attempt benchmark
        result = run_benchmark(dtype)
        return {"success": True, "result": result}
    except RuntimeError as e:
        # GPU-specific errors
        return {"success": False, "error": str(e), "dtype": str(dtype)}
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in benchmark: {e}")
        raise
```

### Device Compatibility

Always support all platforms:

```python
def synchronize_device(device: torch.device):
    """Synchronize device (CUDA, ROCm, or MPS)"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    # CPU needs no sync
```

---

## Testing Guidelines

### Manual Testing

Before submitting, test on your platform:

```bash
# Test all benchmarks
python benchmark.py --all

# Test specific benchmark
python benchmark.py --flops --iterations 50

# Test exports
python benchmark.py --flops --json test.json --csv test.csv

# Verify results
cat test.json | python -m json.tool  # Pretty print
```

### Cross-Platform Testing

If possible, test on multiple platforms:
- ✅ NVIDIA GPU (CUDA)
- ✅ AMD GPU (ROCm)
- ✅ Apple Silicon (MPS)

If you don't have access, mention this in your PR and we'll test.

### Unit Tests (Future)

We're adding pytest-based tests. When available:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_timing.py

# With coverage
pytest --cov=gpu_benchmark tests/
```

---

## Adding New Features

### Adding a New Benchmark

**1. Create benchmark file** (`gpu_benchmark/benchmarks/your_benchmark.py`):

```python
import torch
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DeviceTimer, benchmark_operation


class YourBenchmark:
    """Benchmark for [what it tests]"""

    def __init__(self, device: torch.device, warmup_iterations: int = 10, test_iterations: int = 100):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results = {}

    def run_all(self) -> Dict:
        """Run all tests for this benchmark"""
        print("\n" + "="*60)
        print("YOUR BENCHMARK")
        print("="*60)

        # Implement your benchmark
        self.results["test1"] = self._test_something()

        return self.results

    def _test_something(self) -> Dict:
        """Internal test method"""
        # Your test logic here
        pass

    def get_results(self) -> Dict:
        """Get benchmark results"""
        return self.results

    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("YOUR BENCHMARK SUMMARY")
        print("="*60)
        # Print your summary
        print("="*60)
```

**2. Add to main CLI** (if appropriate):

```python
# In gpu_benchmark/main.py

# Import
from gpu_benchmark.benchmarks.your_benchmark import YourBenchmark

# Add argument
parser.add_argument('--your-bench', action='store_true', help='Run your benchmark')

# Add execution
if args.your_bench or args.all:
    print("\n[X/Y] Running Your Benchmark...")
    results["benchmarks_run"].append("your_bench")
    your_bench = YourBenchmark(device)
    results["your_bench"] = your_bench.run_all()
    your_bench.print_summary()
```

**3. Document in README**:
- Add to features list
- Add usage example
- Explain what it measures

### Adding a New Utility

**1. Create utility file** (`gpu_benchmark/utils/your_util.py`):

```python
"""
Your Utility Module
Brief description of what this module does
"""

from typing import Dict, List


class YourUtility:
    """Utility for [purpose]"""

    def __init__(self, param1: type1):
        self.param1 = param1

    def do_something(self) -> result_type:
        """
        Do something useful

        Args:
            ...

        Returns:
            ...
        """
        pass
```

**2. Export from utils** (`gpu_benchmark/utils/__init__.py`):

```python
from gpu_benchmark.utils.your_util import YourUtility

__all__ = [..., 'YourUtility']
```

**3. Add tests and documentation**

---

## Documentation

### README Updates

When adding features, update README.md:

1. **Features section**: Add to the appropriate subsection
2. **Usage examples**: Show how to use your feature
3. **API documentation**: If adding developer-facing APIs
4. **Troubleshooting**: Add common issues

### Inline Documentation

```python
# Good: Self-documenting code with docstrings
def calculate_roofline_efficiency(
    achieved_gflops: float,
    peak_gflops: float,
    operational_intensity: float,
    peak_bandwidth_gbs: float
) -> float:
    """
    Calculate efficiency using roofline model

    The roofline model determines whether performance is
    limited by compute (FLOPS) or memory bandwidth.

    Args:
        achieved_gflops: Measured performance in GFLOP/s
        peak_gflops: Theoretical peak GFLOP/s
        operational_intensity: FLOP/byte ratio
        peak_bandwidth_gbs: Peak memory bandwidth in GB/s

    Returns:
        Efficiency as percentage (0-100)

    Example:
        >>> calculate_roofline_efficiency(8000, 10000, 50, 500)
        80.0
    """
    ridge_point = peak_gflops / peak_bandwidth_gbs

    if operational_intensity < ridge_point:
        # Memory-bound
        theoretical = operational_intensity * peak_bandwidth_gbs
    else:
        # Compute-bound
        theoretical = peak_gflops

    return (achieved_gflops / theoretical) * 100
```

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (Black + Flake8)
- [ ] All functions have docstrings
- [ ] Type hints added where appropriate
- [ ] Tested on at least one platform
- [ ] No obvious memory leaks
- [ ] Documentation updated (README, docstrings)
- [ ] CHANGELOG.md updated (if significant)
- [ ] Commit messages are clear and descriptive

### Commit Messages

Use clear, descriptive commit messages:

**Format:**
```
<type>: <short summary>

<optional detailed description>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `perf:` Performance improvement
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

**Examples:**
```bash
# Good
git commit -m "feat: Add BF16 support to FLOPS benchmark"
git commit -m "fix: Handle MPS pinned memory error gracefully"
git commit -m "docs: Add roofline analysis examples to README"

# Not ideal
git commit -m "fixed stuff"
git commit -m "updates"
```

### Submitting PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub:
   - Clear title describing the change
   - Reference related issues (#123)
   - Describe what changed and why
   - Include test results (screenshots if applicable)
   - List any breaking changes

**PR Template:**
```markdown
## Description
Brief description of the changes

## Related Issues
Fixes #123

## Changes Made
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- [x] Tested on NVIDIA GPU (RTX 4090)
- [ ] Tested on AMD GPU
- [x] Tested on Apple Silicon (M3 Pro)
- [x] All benchmarks pass
- [x] JSON/CSV export works

## Screenshots (if applicable)
[Add screenshots showing results]

## Checklist
- [x] Code follows style guidelines
- [x] Documentation updated
- [x] CHANGELOG.md updated
- [x] Self-review completed
```

3. **Respond to feedback**: Address reviewer comments promptly

4. **Get approval**: Maintainers will review and merge

---

## Release Process (Maintainers)

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **Major** (x.0.0): Breaking changes
- **Minor** (0.x.0): New features (backward compatible)
- **Patch** (0.0.x): Bug fixes

### Creating a Release

1. Update `CHANGELOG.md`
2. Update version in `setup.py` and `__init__.py`
3. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push --tags`
5. Create GitHub release with notes

---

## Community

### Getting Help

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue
- **Ideas**: Open an Issue with `[Feature Request]`

### Recognition

Contributors will be recognized in:
- `CHANGELOG.md` for their contributions
- GitHub contributors list
- Special thanks in releases

---

## Thank You!

Every contribution, no matter how small, helps make GPU AI Benchmark better for everyone. We appreciate your time and effort!

**Happy benchmarking!** 🚀

---

*This guide is inspired by contributing guides from PyTorch, TensorFlow, and other major open-source projects.*
