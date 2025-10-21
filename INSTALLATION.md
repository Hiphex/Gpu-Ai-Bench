# Installation Guide

Complete installation instructions for all platforms and Python versions.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Python Version Compatibility](#python-version-compatibility)
3. [Windows Installation](#windows-installation)
4. [macOS Installation](#macos-installation)
5. [Linux Installation](#linux-installation)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Standard Installation (Python 3.8-3.12)

```bash
# Clone repository
git clone https://github.com/Hiphex/Gpu-Ai-Bench.git
cd Gpu-Ai-Bench

# Install dependencies
pip install -r requirements.txt

# Run benchmark
python benchmark.py --all
```

### Python 3.13 Installation (Newer Python)

⚠️ **Python 3.13 is very new and PyTorch support is limited. We recommend Python 3.11 or 3.12.**

If you must use Python 3.13, follow these steps:

```bash
# Step 1: Install PyTorch manually (skip torchaudio)
# For CUDA (NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining dependencies
pip install numpy pandas psutil matplotlib

# Step 3: (Optional) Install NVIDIA monitoring (CUDA only)
pip install pynvml

# Step 4: Run benchmark
python benchmark.py --all
```

---

## Python Version Compatibility

| Python Version | Compatibility | Notes |
|----------------|---------------|-------|
| 3.13 | ⚠️ Partial | PyTorch support limited, manual install required |
| 3.12 | ✅ Full | Recommended |
| 3.11 | ✅ Full | Recommended |
| 3.10 | ✅ Full | Stable |
| 3.9 | ✅ Full | Stable |
| 3.8 | ✅ Full | Minimum version |
| 3.7 or older | ❌ Not supported | Upgrade Python |

**Recommended:** Use Python 3.11 or 3.12 for best compatibility.

---

## Windows Installation

### Option 1: Standard (Python 3.8-3.12)

```powershell
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas psutil pynvml matplotlib

# Run benchmark
python benchmark.py --all
```

### Option 2: Python 3.13 (Manual)

```powershell
# Install PyTorch WITHOUT torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install minimal dependencies
pip install -r requirements-minimal.txt

# Run benchmark
python benchmark.py --all
```

### Option 3: CPU-Only (No GPU)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas psutil matplotlib

# Run benchmark (will use CPU)
python benchmark.py --all
```

### Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output (CUDA):
```
PyTorch: 2.x.x+cu121
CUDA Available: True
```

---

## macOS Installation

### Apple Silicon (M1/M2/M3/M4)

```bash
# Install PyTorch with MPS support
pip install torch torchvision

# Install dependencies
pip install numpy pandas psutil matplotlib

# Note: pynvml not needed on Mac (NVIDIA-only)

# Run benchmark
python benchmark.py --all
```

### Intel Mac

```bash
# Install PyTorch
pip install torch torchvision

# Install dependencies
pip install numpy pandas psutil matplotlib

# Run benchmark (CPU only on Intel Macs without eGPU)
python benchmark.py --all
```

### Verify MPS Support

```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

Expected output:
```
MPS Available: True
```

---

## Linux Installation

### Ubuntu/Debian

#### NVIDIA GPU (CUDA)

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install numpy pandas psutil pynvml matplotlib

# Run benchmark
python benchmark.py --all
```

#### AMD GPU (ROCm)

```bash
# Install PyTorch with ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install numpy pandas psutil matplotlib

# Note: AMD power monitoring uses rocm-smi (should be pre-installed)

# Run benchmark
python benchmark.py --all
```

#### CPU Only

```bash
pip install torch torchvision
pip install numpy pandas psutil matplotlib

python benchmark.py --all
```

### RHEL/CentOS/Fedora

Same as Ubuntu, but you may need to install system dependencies first:

```bash
# Install system packages
sudo dnf install python3-devel gcc

# Then follow Ubuntu instructions above
```

---

## Troubleshooting

### Error: "No matching distribution found for torchaudio"

**Cause:** Python 3.13 is too new, PyTorch doesn't have full support yet.

**Solutions:**

**Option A - Downgrade Python (Recommended):**
```bash
# Install Python 3.12 or 3.11
# Then install normally
pip install -r requirements.txt
```

**Option B - Skip torchaudio (Our benchmark doesn't need it):**
```bash
# Install PyTorch without audio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install rest manually
pip install numpy pandas psutil pynvml matplotlib
```

### Error: "Could not find a version that satisfies the requirement torch"

**Cause:** No pre-built wheel for your Python/OS combination.

**Solution:** Check [PyTorch website](https://pytorch.org/get-started/locally/) for correct install command.

### Error: "CUDA not available" (on NVIDIA GPU)

**Cause:** PyTorch CPU version installed instead of CUDA version.

**Solution:** Reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Error: "MPS not available" (on Apple Silicon)

**Cause:** Old PyTorch version or OS version.

**Solution:**
```bash
# Update PyTorch
pip install --upgrade torch torchvision

# Verify macOS version (need 12.3+)
sw_vers
```

### Error: "pynvml not available"

**Cause:** pynvml isn't installed or you're not on NVIDIA GPU.

**Solutions:**
- On NVIDIA GPU: `pip install pynvml`
- On AMD/Apple: This is normal, pynvml is NVIDIA-only (benchmark will skip it)

### Error: "torchvision not available"

**Cause:** torchvision not installed (needed for ResNet benchmark).

**Solution:**
```bash
pip install torchvision
```

### Low Performance on Mac

**Causes:**
1. Running on battery power (40-60% slower)
2. Thermal throttling
3. Other apps using GPU

**Solutions:**
- Plug in to power
- Close other GPU-intensive apps
- Run with thermal monitoring: `python benchmark_enhanced.py --all --thermal`

### ImportError on Windows

**Cause:** Missing Visual C++ redistributables.

**Solution:** Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## Platform-Specific Commands

### Check Python Version

```bash
python --version
```

### Check if GPU is detected

**NVIDIA:**
```bash
nvidia-smi
```

**AMD:**
```bash
rocm-smi
```

**Apple:**
```bash
system_profiler SPDisplaysDataType | grep Chipset
```

### Verify PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
```

---

## Virtual Environment (Recommended)

Using a virtual environment prevents dependency conflicts:

### Create Virtual Environment

```bash
# Python 3 venv
python -m venv gpu-bench-env

# Activate (Linux/Mac)
source gpu-bench-env/bin/activate

# Activate (Windows)
gpu-bench-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run benchmark
python benchmark.py --all

# Deactivate when done
deactivate
```

### Anaconda/Miniconda

```bash
# Create environment
conda create -n gpu-bench python=3.12

# Activate
conda activate gpu-bench

# Install PyTorch (CUDA)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other deps
pip install numpy pandas psutil pynvml matplotlib

# Run benchmark
python benchmark.py --all
```

---

## Minimal Installation (Core Features Only)

If you only want basic benchmarks without ResNet/BERT:

```bash
# Install PyTorch
pip install torch

# Install minimal deps
pip install numpy pandas psutil

# Run basic benchmarks (skip --resnet --bert)
python benchmark.py --flops --memory --power
```

---

## Docker Installation (Advanced)

Coming soon - pre-built Docker images with all dependencies.

---

## Quick Install Scripts

### Windows (PowerShell)

Save as `install.ps1`:

```powershell
# GPU AI Benchmark - Windows Installer
Write-Host "Installing GPU AI Benchmark..." -ForegroundColor Green

# Check Python version
$pythonVersion = python --version
Write-Host "Python version: $pythonVersion"

# Install PyTorch with CUDA
Write-Host "Installing PyTorch with CUDA..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install numpy pandas psutil pynvml matplotlib

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Run: python benchmark.py --all" -ForegroundColor Cyan
```

### Linux/macOS (Bash)

Save as `install.sh`:

```bash
#!/bin/bash
# GPU AI Benchmark - Unix Installer

echo "Installing GPU AI Benchmark..."

# Check Python version
echo "Python version:"
python3 --version

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing for macOS..."
    pip3 install torch torchvision
else
    echo "Installing for Linux with CUDA..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install numpy pandas psutil matplotlib

# Linux only: pynvml
if [[ "$OSTYPE" != "darwin"* ]]; then
    pip3 install pynvml
fi

echo "Installation complete!"
echo "Run: python3 benchmark.py --all"
```

Make executable:
```bash
chmod +x install.sh
./install.sh
```

---

## Getting Help

If you encounter issues:

1. **Check Python version**: `python --version` (need 3.8-3.13)
2. **Verify PyTorch**: Run verification script above
3. **Read error messages**: Most errors indicate missing packages
4. **Check GPU**: Use `nvidia-smi`, `rocm-smi`, or macOS System Report
5. **Use minimal install**: Try `requirements-minimal.txt` if full install fails

Still stuck? Open an issue: https://github.com/Hiphex/Gpu-Ai-Bench/issues

Include:
- Operating system and version
- Python version
- GPU model
- Full error message
- Output of `pip list`
