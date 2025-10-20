"""
GPU Performance Baselines Database

Known performance baselines for comparison
Data sourced from MLPerf, Lambda benchmarks, and vendor specs
"""

from typing import Dict, Optional, List


# NVIDIA GPU Baselines
NVIDIA_BASELINES = {
    # RTX 40 Series (Ada Lovelace)
    "RTX 4090": {
        "vendor": "NVIDIA",
        "architecture": "Ada Lovelace",
        "compute_capability": "8.9",
        "fp32_tflops": 82.6,
        "fp16_tflops": 165.2,
        "tensor_tflops": 660.6,  # FP16 with Tensor Cores
        "memory_gb": 24,
        "memory_bandwidth_gbps": 1008,
        "tdp_watts": 450,
        "resnet50_training_img_sec": 1200,  # Approximate
        "bert_training_seq_sec": 180,  # Approximate
    },
    "RTX 4080": {
        "vendor": "NVIDIA",
        "architecture": "Ada Lovelace",
        "compute_capability": "8.9",
        "fp32_tflops": 48.7,
        "fp16_tflops": 97.4,
        "tensor_tflops": 389.0,
        "memory_gb": 16,
        "memory_bandwidth_gbps": 736,
        "tdp_watts": 320,
        "resnet50_training_img_sec": 850,
        "bert_training_seq_sec": 120,
    },
    "RTX 4070 Ti": {
        "vendor": "NVIDIA",
        "architecture": "Ada Lovelace",
        "compute_capability": "8.9",
        "fp32_tflops": 40.1,
        "fp16_tflops": 80.2,
        "tensor_tflops": 321.0,
        "memory_gb": 12,
        "memory_bandwidth_gbps": 504,
        "tdp_watts": 285,
        "resnet50_training_img_sec": 700,
        "bert_training_seq_sec": 100,
    },

    # RTX 30 Series (Ampere)
    "RTX 3090": {
        "vendor": "NVIDIA",
        "architecture": "Ampere",
        "compute_capability": "8.6",
        "fp32_tflops": 35.6,
        "fp16_tflops": 71,
        "tensor_tflops": 285,
        "memory_gb": 24,
        "memory_bandwidth_gbps": 936,
        "tdp_watts": 350,
        "resnet50_training_img_sec": 800,
        "bert_training_seq_sec": 110,
    },
    "RTX 3080": {
        "vendor": "NVIDIA",
        "architecture": "Ampere",
        "compute_capability": "8.6",
        "fp32_tflops": 29.8,
        "fp16_tflops": 59.5,
        "tensor_tflops": 238,
        "memory_gb": 10,
        "memory_bandwidth_gbps": 760,
        "tdp_watts": 320,
        "resnet50_training_img_sec": 650,
        "bert_training_seq_sec": 90,
    },

    # A100 (Data Center)
    "A100 80GB": {
        "vendor": "NVIDIA",
        "architecture": "Ampere",
        "compute_capability": "8.0",
        "fp32_tflops": 19.5,
        "fp16_tflops": 312,
        "tensor_tflops": 624,  # FP16 with Tensor Cores
        "memory_gb": 80,
        "memory_bandwidth_gbps": 2039,
        "tdp_watts": 400,
        "resnet50_training_img_sec": 1500,
        "bert_training_seq_sec": 250,
    },
    "A100 40GB": {
        "vendor": "NVIDIA",
        "architecture": "Ampere",
        "compute_capability": "8.0",
        "fp32_tflops": 19.5,
        "fp16_tflops": 312,
        "tensor_tflops": 624,
        "memory_gb": 40,
        "memory_bandwidth_gbps": 1555,
        "tdp_watts": 400,
        "resnet50_training_img_sec": 1400,
        "bert_training_seq_sec": 240,
    },

    # H100 (Hopper - Data Center)
    "H100 80GB": {
        "vendor": "NVIDIA",
        "architecture": "Hopper",
        "compute_capability": "9.0",
        "fp32_tflops": 67,
        "fp16_tflops": 1979,
        "tensor_tflops": 3958,  # FP8 with Tensor Cores
        "memory_gb": 80,
        "memory_bandwidth_gbps": 3352,
        "tdp_watts": 700,
        "resnet50_training_img_sec": 3000,
        "bert_training_seq_sec": 500,
    },
}

# AMD GPU Baselines
AMD_BASELINES = {
    "RX 7900 XTX": {
        "vendor": "AMD",
        "architecture": "RDNA 3",
        "fp32_tflops": 61,
        "fp16_tflops": 122,
        "memory_gb": 24,
        "memory_bandwidth_gbps": 960,
        "tdp_watts": 355,
        "resnet50_training_img_sec": 600,
        "bert_training_seq_sec": 80,
    },
    "RX 7900 XT": {
        "vendor": "AMD",
        "architecture": "RDNA 3",
        "fp32_tflops": 51.5,
        "fp16_tflops": 103,
        "memory_gb": 20,
        "memory_bandwidth_gbps": 800,
        "tdp_watts": 315,
        "resnet50_training_img_sec": 500,
        "bert_training_seq_sec": 70,
    },
    "MI250X": {
        "vendor": "AMD",
        "architecture": "CDNA 2",
        "fp32_tflops": 47.9,
        "fp16_tflops": 383,
        "memory_gb": 128,
        "memory_bandwidth_gbps": 3277,
        "tdp_watts": 560,
        "resnet50_training_img_sec": 2000,
        "bert_training_seq_sec": 300,
    },
    "MI300X": {
        "vendor": "AMD",
        "architecture": "CDNA 3",
        "fp32_tflops": 163,
        "fp16_tflops": 1307,
        "memory_gb": 192,
        "memory_bandwidth_gbps": 5300,
        "tdp_watts": 750,
        "resnet50_training_img_sec": 3500,
        "bert_training_seq_sec": 550,
    },
}

# Apple Silicon Baselines
APPLE_BASELINES = {
    "M1": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 8,
        "fp32_tflops": 2.6,
        "fp16_tflops": 5.2,
        "memory_gb": 16,  # Unified memory
        "memory_bandwidth_gbps": 68.25,
        "tdp_watts": 15,  # Approximate for whole SoC
        "resnet50_training_img_sec": 80,
        "bert_training_seq_sec": 12,
    },
    "M1 Pro": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 16,
        "fp32_tflops": 5.2,
        "fp16_tflops": 10.4,
        "memory_gb": 32,
        "memory_bandwidth_gbps": 200,
        "tdp_watts": 30,
        "resnet50_training_img_sec": 150,
        "bert_training_seq_sec": 20,
    },
    "M1 Max": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 32,
        "fp32_tflops": 10.4,
        "fp16_tflops": 20.8,
        "memory_gb": 64,
        "memory_bandwidth_gbps": 400,
        "tdp_watts": 60,
        "resnet50_training_img_sec": 250,
        "bert_training_seq_sec": 35,
    },
    "M2": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 10,
        "fp32_tflops": 3.6,
        "fp16_tflops": 7.2,
        "memory_gb": 24,
        "memory_bandwidth_gbps": 100,
        "tdp_watts": 20,
        "resnet50_training_img_sec": 100,
        "bert_training_seq_sec": 15,
    },
    "M3": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 10,
        "fp32_tflops": 4.1,
        "fp16_tflops": 8.2,
        "memory_gb": 24,
        "memory_bandwidth_gbps": 100,
        "tdp_watts": 20,
        "resnet50_training_img_sec": 110,
        "bert_training_seq_sec": 16,
    },
    "M3 Pro": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 18,
        "fp32_tflops": 7.4,
        "fp16_tflops": 14.8,
        "memory_gb": 36,
        "memory_bandwidth_gbps": 150,
        "tdp_watts": 35,
        "resnet50_training_img_sec": 180,
        "bert_training_seq_sec": 25,
    },
    "M3 Max": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 40,
        "fp32_tflops": 16.4,
        "fp16_tflops": 32.8,
        "memory_gb": 128,
        "memory_bandwidth_gbps": 400,
        "tdp_watts": 80,
        "resnet50_training_img_sec": 350,
        "bert_training_seq_sec": 50,
    },
    "M4": {
        "vendor": "Apple",
        "architecture": "Apple Silicon",
        "gpu_cores": 10,
        "fp32_tflops": 4.5,
        "fp16_tflops": 9.0,
        "memory_gb": 24,
        "memory_bandwidth_gbps": 120,
        "tdp_watts": 22,
        "resnet50_training_img_sec": 120,
        "bert_training_seq_sec": 18,
    },
}

# Combined database
GPU_BASELINES = {
    **NVIDIA_BASELINES,
    **AMD_BASELINES,
    **APPLE_BASELINES,
}


def get_baseline(gpu_name: str) -> Optional[Dict]:
    """
    Get baseline performance data for a GPU

    Args:
        gpu_name: GPU model name (e.g., "RTX 4090", "M3 Pro")

    Returns:
        Dict with baseline data, or None if not found
    """
    return GPU_BASELINES.get(gpu_name)


def find_closest_baseline(fp32_tflops: float, memory_gb: float, vendor: str = None) -> Optional[tuple]:
    """
    Find closest matching GPU baseline based on specs

    Args:
        fp32_tflops: Measured FP32 TFLOPS
        memory_gb: Total memory in GB
        vendor: Optional vendor filter ("NVIDIA", "AMD", "Apple")

    Returns:
        Tuple of (gpu_name, baseline_dict, similarity_score) or None
    """
    candidates = []

    for gpu_name, baseline in GPU_BASELINES.items():
        if vendor and baseline.get("vendor") != vendor:
            continue

        # Calculate similarity score (lower is better)
        tflops_diff = abs(baseline.get("fp32_tflops", 0) - fp32_tflops)
        memory_diff = abs(baseline.get("memory_gb", 0) - memory_gb)

        # Normalize differences
        tflops_score = tflops_diff / max(baseline.get("fp32_tflops", 1), fp32_tflops)
        memory_score = memory_diff / max(baseline.get("memory_gb", 1), memory_gb)

        similarity_score = tflops_score + memory_score

        candidates.append((gpu_name, baseline, similarity_score))

    if not candidates:
        return None

    # Return closest match
    candidates.sort(key=lambda x: x[2])
    return candidates[0]


def compare_to_baseline(measured_results: Dict, baseline_name: str) -> Dict:
    """
    Compare measured results to a baseline

    Args:
        measured_results: Dict with measured benchmark results
        baseline_name: Name of GPU to compare against

    Returns:
        Dict with comparison data
    """
    baseline = get_baseline(baseline_name)
    if not baseline:
        return {"error": f"Baseline '{baseline_name}' not found"}

    comparison = {
        "baseline_gpu": baseline_name,
        "comparisons": {}
    }

    # Compare FLOPS
    if "fp32_tflops" in measured_results and "fp32_tflops" in baseline:
        ratio = measured_results["fp32_tflops"] / baseline["fp32_tflops"]
        comparison["comparisons"]["fp32_tflops"] = {
            "measured": measured_results["fp32_tflops"],
            "baseline": baseline["fp32_tflops"],
            "ratio": ratio,
            "percent_of_baseline": ratio * 100
        }

    # Compare memory bandwidth
    if "memory_bandwidth_gbps" in measured_results and "memory_bandwidth_gbps" in baseline:
        ratio = measured_results["memory_bandwidth_gbps"] / baseline["memory_bandwidth_gbps"]
        comparison["comparisons"]["memory_bandwidth_gbps"] = {
            "measured": measured_results["memory_bandwidth_gbps"],
            "baseline": baseline["memory_bandwidth_gbps"],
            "ratio": ratio,
            "percent_of_baseline": ratio * 100
        }

    # Compare ResNet50 training
    if "resnet50_training_img_sec" in measured_results and "resnet50_training_img_sec" in baseline:
        ratio = measured_results["resnet50_training_img_sec"] / baseline["resnet50_training_img_sec"]
        comparison["comparisons"]["resnet50_training"] = {
            "measured": measured_results["resnet50_training_img_sec"],
            "baseline": baseline["resnet50_training_img_sec"],
            "ratio": ratio,
            "percent_of_baseline": ratio * 100
        }

    return comparison


def list_baselines_by_vendor(vendor: str = None) -> List[str]:
    """
    List all available baselines, optionally filtered by vendor

    Args:
        vendor: Optional vendor filter ("NVIDIA", "AMD", "Apple")

    Returns:
        List of GPU names
    """
    if vendor:
        return [name for name, data in GPU_BASELINES.items()
                if data.get("vendor") == vendor]
    else:
        return list(GPU_BASELINES.keys())


if __name__ == "__main__":
    # Demo usage
    print("Available baselines:")
    print(f"  NVIDIA: {len(list_baselines_by_vendor('NVIDIA'))}")
    print(f"  AMD: {len(list_baselines_by_vendor('AMD'))}")
    print(f"  Apple: {len(list_baselines_by_vendor('Apple'))}")

    print("\nRTX 4090 baseline:")
    print(get_baseline("RTX 4090"))

    print("\nFinding closest baseline for 7.4 TFLOPS, 36 GB:")
    closest = find_closest_baseline(7.4, 36)
    if closest:
        print(f"  Closest match: {closest[0]} (similarity: {closest[2]:.3f})")
