from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpu-ai-benchmark",
    version="1.0.0",
    author="GPU AI Bench Team",
    description="Comprehensive GPU benchmarking tool for AI workloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hiphex/Gpu-Ai-Bench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Benchmark",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "nvidia": ["pynvml>=11.5.0"],
    },
    entry_points={
        "console_scripts": [
            "gpu-benchmark=gpu_benchmark.main:main",
        ],
    },
)
