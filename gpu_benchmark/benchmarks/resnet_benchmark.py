"""
ResNet50 Benchmark - Real-world computer vision workload
Measures training and inference throughput for ResNet50 (MLPerf standard)
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Optional
import warnings

try:
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not available - ResNet benchmark will be limited")


class ResNet50Benchmark:
    """
    ResNet50 benchmark for training and inference throughput

    Metrics:
    - Training: images/second
    - Inference: images/second, latency (ms)
    - Mixed-precision (AMP) vs FP32
    """

    def __init__(self, device: torch.device, batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.results = {}

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for ResNet benchmark: pip install torchvision")

    def _create_model(self) -> nn.Module:
        """Create ResNet50 model"""
        model = resnet50(weights=None)  # Random init for consistent benchmarking
        model = model.to(self.device)
        return model

    def _create_dummy_batch(self, batch_size: int) -> tuple:
        """Create dummy image batch (batch_size, 3, 224, 224) and labels"""
        images = torch.randn(batch_size, 3, 224, 224, device=self.device)
        labels = torch.randint(0, 1000, (batch_size,), device=self.device)
        return images, labels

    def _synchronize(self):
        """Synchronize device"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()

    def benchmark_inference_throughput(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark inference throughput

        Returns:
            Dict with images/sec, latency, and statistics
        """
        model = self._create_model()
        model.eval()

        images, _ = self._create_dummy_batch(self.batch_size)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(images)

        self._synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(images)
                self._synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate metrics
        avg_latency_ms = sum(latencies) / len(latencies)
        throughput = (self.batch_size / avg_latency_ms) * 1000  # images/sec

        return {
            "throughput_images_per_sec": throughput,
            "latency_ms": avg_latency_ms,
            "latency_per_image_ms": avg_latency_ms / self.batch_size,
            "batch_size": self.batch_size,
            "num_iterations": num_iterations,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    def benchmark_training_throughput(self, num_iterations: int = 50, use_amp: bool = False) -> Dict:
        """
        Benchmark training throughput

        Args:
            num_iterations: Number of training iterations
            use_amp: Use Automatic Mixed Precision (AMP)

        Returns:
            Dict with images/sec and training statistics
        """
        model = self._create_model()
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        images, labels = self._create_dummy_batch(self.batch_size)

        # Setup AMP if requested (CUDA only)
        use_amp = use_amp and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Warmup
        for _ in range(5):
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self._synchronize()

        # Benchmark
        iteration_times = []
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            iter_start = time.perf_counter()

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self._synchronize()
            iter_end = time.perf_counter()
            iteration_times.append(iter_end - iter_start)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate metrics
        total_images = num_iterations * self.batch_size
        throughput = total_images / total_time
        avg_iter_time = sum(iteration_times) / len(iteration_times)

        return {
            "throughput_images_per_sec": throughput,
            "avg_iteration_time_sec": avg_iter_time,
            "total_time_sec": total_time,
            "batch_size": self.batch_size,
            "num_iterations": num_iterations,
            "total_images": total_images,
            "use_amp": use_amp,
            "min_iteration_time_sec": min(iteration_times),
            "max_iteration_time_sec": max(iteration_times),
        }

    def run_all(self) -> Dict:
        """Run all ResNet50 benchmarks"""
        print("\n" + "="*60)
        print("ResNet50 Benchmark - Real-world Computer Vision Workload")
        print("="*60)

        # Inference benchmark
        print(f"\n[1/3] Running inference benchmark (batch_size={self.batch_size})...")
        self.results["inference"] = self.benchmark_inference_throughput()
        print(f"  → Throughput: {self.results['inference']['throughput_images_per_sec']:.2f} images/sec")
        print(f"  → Latency: {self.results['inference']['latency_ms']:.2f} ms")

        # Training FP32
        print(f"\n[2/3] Running training benchmark FP32 (batch_size={self.batch_size})...")
        self.results["training_fp32"] = self.benchmark_training_throughput(use_amp=False)
        print(f"  → Throughput: {self.results['training_fp32']['throughput_images_per_sec']:.2f} images/sec")

        # Training AMP (CUDA only)
        if self.device.type == 'cuda':
            print(f"\n[3/3] Running training benchmark AMP (batch_size={self.batch_size})...")
            self.results["training_amp"] = self.benchmark_training_throughput(use_amp=True)
            print(f"  → Throughput: {self.results['training_amp']['throughput_images_per_sec']:.2f} images/sec")

            speedup = (self.results['training_amp']['throughput_images_per_sec'] /
                      self.results['training_fp32']['throughput_images_per_sec'])
            print(f"  → AMP Speedup: {speedup:.2f}x over FP32")
        else:
            print(f"\n[3/3] Skipping AMP benchmark (only available on CUDA)")

        print("\n" + "="*60)

        return self.results


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    benchmark = ResNet50Benchmark(device=device, batch_size=32)
    results = benchmark.run_all()

    print("\nResults Summary:")
    print(f"Inference: {results['inference']['throughput_images_per_sec']:.2f} images/sec")
    print(f"Training FP32: {results['training_fp32']['throughput_images_per_sec']:.2f} images/sec")
    if 'training_amp' in results:
        print(f"Training AMP: {results['training_amp']['throughput_images_per_sec']:.2f} images/sec")
