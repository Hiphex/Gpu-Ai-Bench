"""
Results Export Module
Exports benchmark results to JSON and CSV formats
"""

import json
import csv
from datetime import datetime
from typing import Dict
import os


class ResultExporter:
    """Export benchmark results to various formats"""

    def __init__(self, results: Dict, output_dir: str = "results"):
        self.results = results
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def export_json(self, filename: str = None) -> str:
        """Export results to JSON file"""
        if filename is None:
            filename = f"gpu_benchmark_{self.timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults exported to JSON: {filepath}")
        return filepath

    def export_csv(self, filename: str = None) -> str:
        """Export results to CSV file"""
        if filename is None:
            filename = f"gpu_benchmark_{self.timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        # Flatten the results dictionary for CSV export
        flat_results = self._flatten_results(self.results)

        with open(filepath, 'w', newline='') as f:
            if flat_results:
                writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
                writer.writeheader()
                writer.writerows(flat_results)
            else:
                # If no flat results, write the keys and values directly
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for key, value in self._flatten_dict(self.results).items():
                    writer.writerow([key, value])

        print(f"Results exported to CSV: {filepath}")
        return filepath

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten a nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _flatten_results(self, results: Dict) -> list:
        """Flatten results for CSV export"""
        rows = []

        # Add GPU info as a row
        if "gpu_info" in results:
            row = {"category": "GPU Info"}
            row.update({k: v for k, v in results["gpu_info"].items()})
            rows.append(row)

        # Add FLOPS results
        if "flops" in results:
            for precision, data in results["flops"].items():
                row = {
                    "category": "FLOPS",
                    "precision": precision,
                    "average_tflops": data.get("average_tflops", 0)
                }
                rows.append(row)

        # Add memory results
        if "memory" in results:
            mem_info = results["memory"].get("memory_info", {})
            bandwidth = results["memory"].get("bandwidth", {})
            row = {
                "category": "Memory",
                "total_memory_gb": mem_info.get("total_gb", 0),
                "peak_bandwidth_gbs": bandwidth.get("peak_bandwidth_gbs", 0)
            }
            rows.append(row)

        # Add power results
        if "power" in results and results["power"].get("available", True):
            power_info = results["power"].get("power_info", {})
            stress_test = results["power"].get("stress_test", {})
            row = {
                "category": "Power",
                "power_limit_w": power_info.get("power_limit_w", 0),
                "idle_power_w": power_info.get("current_power_w", 0),
                "stress_avg_power_w": stress_test.get("avg_power_w", 0),
                "stress_max_power_w": stress_test.get("max_power_w", 0)
            }
            rows.append(row)

        # Add workload results
        if "workloads" in results:
            workloads = results["workloads"]
            if "convolution" in workloads:
                row = {
                    "category": "Workload",
                    "type": "Convolution",
                    "throughput": workloads["convolution"].get("average_throughput_img_per_sec", 0),
                    "unit": "img/s"
                }
                rows.append(row)

            if "attention" in workloads:
                row = {
                    "category": "Workload",
                    "type": "Attention",
                    "throughput": workloads["attention"].get("average_throughput_tokens_per_sec", 0),
                    "unit": "tokens/s"
                }
                rows.append(row)

        return rows

    def print_summary_report(self):
        """Print a comprehensive summary report"""
        print("\n" + "="*80)
        print(" "*25 + "GPU BENCHMARK REPORT")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # GPU Info
        if "gpu_info" in self.results:
            print("\nGPU INFORMATION:")
            print("-" * 80)
            gpu_info = self.results["gpu_info"]
            print(f"  Name: {gpu_info.get('name', 'N/A')}")
            print(f"  Vendor: {gpu_info.get('vendor', 'N/A')}")
            print(f"  Memory: {gpu_info.get('total_memory_gb', 0):.2f} GB")
            print(f"  Compute Capability: {gpu_info.get('compute_capability', 'N/A')}")
            print(f"  Backend: {gpu_info.get('backend', 'N/A')}")

        # FLOPS
        if "flops" in self.results:
            print("\nFLOPS PERFORMANCE:")
            print("-" * 80)
            for precision, data in self.results["flops"].items():
                print(f"  {precision.upper()}: {data.get('average_tflops', 0):.2f} TFLOPS")

        # Memory
        if "memory" in self.results:
            print("\nMEMORY PERFORMANCE:")
            print("-" * 80)
            mem = self.results["memory"]
            if "memory_info" in mem:
                print(f"  Total Memory: {mem['memory_info'].get('total_gb', 0):.2f} GB")
            if "bandwidth" in mem:
                print(f"  Peak Bandwidth: {mem['bandwidth'].get('peak_bandwidth_gbs', 0):.2f} GB/s")

        # Power
        if "power" in self.results and self.results["power"].get("available", True):
            print("\nPOWER CONSUMPTION:")
            print("-" * 80)
            power = self.results["power"]
            if "power_info" in power:
                print(f"  Power Limit: {power['power_info'].get('power_limit_w', 0):.2f} W")
                print(f"  Idle Power: {power['power_info'].get('current_power_w', 0):.2f} W")
            if "stress_test" in power:
                print(f"  Stress Test Average: {power['stress_test'].get('avg_power_w', 0):.2f} W")
                print(f"  Stress Test Max: {power['stress_test'].get('max_power_w', 0):.2f} W")

        # Workloads
        if "workloads" in self.results:
            print("\nAI WORKLOAD PERFORMANCE:")
            print("-" * 80)
            workloads = self.results["workloads"]
            if "convolution" in workloads:
                print(f"  Convolution: {workloads['convolution'].get('average_throughput_img_per_sec', 0):.2f} img/s")
            if "attention" in workloads:
                print(f"  Attention: {workloads['attention'].get('average_throughput_tokens_per_sec', 0):.2f} tokens/s")

        print("\n" + "="*80)
