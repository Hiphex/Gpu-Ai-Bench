# Windows Python 3.13 Quick Fix

## The Problem

Python 3.13 is too new - PyTorch doesn't have full support for it yet. Specifically, `torchaudio` isn't available.

**Good news:** Our GPU benchmark doesn't need `torchaudio`! We only need `torch` and `torchvision`.

## The Solution (Choose One)

### Option 1: Install Without torchaudio (Recommended)

Open PowerShell or Command Prompt and run:

```powershell
# Step 1: Install PyTorch with CUDA (skip torchaudio)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install other dependencies manually
pip install numpy pandas psutil matplotlib pynvml

# Step 3: Run the benchmark
python benchmark.py --all
```

**That's it!** The benchmark will work perfectly without torchaudio.

### Option 2: Downgrade to Python 3.12 (Better Long-term)

1. Download Python 3.12 from: https://www.python.org/downloads/
2. Install it
3. Then run:

```powershell
pip install -r requirements.txt
python benchmark.py --all
```

---

## For Your Specific Error

You saw:
```
ERROR: Could not find a version that satisfies the requirement torchaudio (from versions: none)
ERROR: No matching distribution found for torchaudio
```

This happened because PyTorch doesn't have `torchaudio` for Python 3.13 yet.

**Fix:**

```powershell
# Install just torch and torchvision (skip torchaudio)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest
pip install numpy pandas psutil matplotlib pynvml

# You're ready!
python benchmark.py --all
```

---

## Verify It's Working

Run this to check:

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

You should see:
```
PyTorch: 2.x.x+cu121
CUDA: True
```

---

## Still Having Issues?

### Error: "numpy.dtype size changed"

```powershell
pip install --upgrade numpy
```

### Error: "CUDA not available"

You might have installed CPU version. Reinstall with:

```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Error: "No module named 'torchvision'"

```powershell
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Why Not Use Python 3.12?

Python 3.13 was just released in October 2024 and many packages haven't caught up yet. Python 3.12 is:
- Fully supported by PyTorch
- More stable
- Better tested
- **Recommended for ML/AI work**

But if you want to stay on 3.13, the commands above will work fine!

---

## Quick Copy-Paste Solution

Open PowerShell and paste this entire block:

```powershell
Write-Host "Installing GPU AI Benchmark for Python 3.13..." -ForegroundColor Green
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas psutil matplotlib pynvml
Write-Host "Installation complete! Run: python benchmark.py --all" -ForegroundColor Cyan
```

That's it! 🚀
