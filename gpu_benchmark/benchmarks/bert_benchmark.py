"""
BERT Benchmark - Real-world NLP workload
Measures training and inference throughput for BERT-Base (MLPerf standard)
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Optional
import warnings


class SimpleBERTModel(nn.Module):
    """
    Simplified BERT-like model for benchmarking

    Similar architecture to BERT-Base:
    - 12 transformer layers
    - 768 hidden size
    - 12 attention heads
    - ~110M parameters
    """

    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12, max_seq_length: int = 512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification

        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        # Embeddings
        token_embeddings = self.embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = token_embeddings + position_embeddings

        # Transformer
        hidden_states = self.transformer(embeddings)

        # Classification (use [CLS] token, first token)
        cls_hidden = hidden_states[:, 0, :]
        logits = self.classifier(cls_hidden)

        return logits


class BERTBenchmark:
    """
    BERT benchmark for training and inference throughput

    Metrics:
    - Training: sequences/second
    - Inference: sequences/second, latency (ms), time-to-first-token (TTFT)
    - Mixed-precision (AMP) vs FP32
    """

    def __init__(self, device: torch.device, batch_size: int = 8, seq_length: int = 512):
        self.device = device
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.results = {}

    def _create_model(self) -> nn.Module:
        """Create BERT-Base model"""
        model = SimpleBERTModel(
            vocab_size=30522,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            max_seq_length=512
        )
        model = model.to(self.device)
        return model

    def _create_dummy_batch(self, batch_size: int) -> tuple:
        """Create dummy token batch and labels"""
        input_ids = torch.randint(0, 30522, (batch_size, self.seq_length), device=self.device)
        labels = torch.randint(0, 2, (batch_size,), device=self.device)
        return input_ids, labels

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
            Dict with sequences/sec, latency, and TTFT
        """
        model = self._create_model()
        model.eval()

        input_ids, _ = self._create_dummy_batch(self.batch_size)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)

        self._synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(input_ids)
                self._synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate metrics
        avg_latency_ms = sum(latencies) / len(latencies)
        throughput = (self.batch_size / avg_latency_ms) * 1000  # sequences/sec

        # TTFT approximation (first layer forward pass time)
        # In real LLM inference, TTFT is time until first token generation
        ttft_ms = avg_latency_ms / 12  # Approximate (12 layers)

        return {
            "throughput_sequences_per_sec": throughput,
            "latency_ms": avg_latency_ms,
            "latency_per_sequence_ms": avg_latency_ms / self.batch_size,
            "ttft_ms": ttft_ms,  # Time to first token approximation
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
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
            Dict with sequences/sec and training statistics
        """
        model = self._create_model()
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        input_ids, labels = self._create_dummy_batch(self.batch_size)

        # Setup AMP if requested (CUDA only)
        use_amp = use_amp and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Warmup
        for _ in range(5):
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids)
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
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self._synchronize()
            iter_end = time.perf_counter()
            iteration_times.append(iter_end - iter_start)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate metrics
        total_sequences = num_iterations * self.batch_size
        throughput = total_sequences / total_time
        avg_iter_time = sum(iteration_times) / len(iteration_times)

        return {
            "throughput_sequences_per_sec": throughput,
            "avg_iteration_time_sec": avg_iter_time,
            "total_time_sec": total_time,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "num_iterations": num_iterations,
            "total_sequences": total_sequences,
            "use_amp": use_amp,
            "min_iteration_time_sec": min(iteration_times),
            "max_iteration_time_sec": max(iteration_times),
        }

    def run_all(self) -> Dict:
        """Run all BERT benchmarks"""
        print("\n" + "="*60)
        print("BERT Benchmark - Real-world NLP Workload")
        print(f"Model: BERT-Base (12 layers, 768 hidden, ~110M params)")
        print("="*60)

        # Inference benchmark
        print(f"\n[1/3] Running inference benchmark (batch_size={self.batch_size}, seq_len={self.seq_length})...")
        self.results["inference"] = self.benchmark_inference_throughput()
        print(f"  → Throughput: {self.results['inference']['throughput_sequences_per_sec']:.2f} sequences/sec")
        print(f"  → Latency: {self.results['inference']['latency_ms']:.2f} ms")
        print(f"  → TTFT (approx): {self.results['inference']['ttft_ms']:.2f} ms")

        # Training FP32
        print(f"\n[2/3] Running training benchmark FP32 (batch_size={self.batch_size}, seq_len={self.seq_length})...")
        self.results["training_fp32"] = self.benchmark_training_throughput(use_amp=False)
        print(f"  → Throughput: {self.results['training_fp32']['throughput_sequences_per_sec']:.2f} sequences/sec")

        # Training AMP (CUDA only)
        if self.device.type == 'cuda':
            print(f"\n[3/3] Running training benchmark AMP (batch_size={self.batch_size}, seq_len={self.seq_length})...")
            self.results["training_amp"] = self.benchmark_training_throughput(use_amp=True)
            print(f"  → Throughput: {self.results['training_amp']['throughput_sequences_per_sec']:.2f} sequences/sec")

            speedup = (self.results['training_amp']['throughput_sequences_per_sec'] /
                      self.results['training_fp32']['throughput_sequences_per_sec'])
            print(f"  → AMP Speedup: {speedup:.2f}x over FP32")
        else:
            print(f"\n[3/3] Skipping AMP benchmark (only available on CUDA)")

        print("\n" + "="*60)

        return self.results


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    benchmark = BERTBenchmark(device=device, batch_size=8, seq_length=512)
    results = benchmark.run_all()

    print("\nResults Summary:")
    print(f"Inference: {results['inference']['throughput_sequences_per_sec']:.2f} sequences/sec")
    print(f"Training FP32: {results['training_fp32']['throughput_sequences_per_sec']:.2f} sequences/sec")
    if 'training_amp' in results:
        print(f"Training AMP: {results['training_amp']['throughput_sequences_per_sec']:.2f} sequences/sec")
