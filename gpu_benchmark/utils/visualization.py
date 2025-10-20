"""
Visualization utilities for benchmark results
Creates charts and plots for performance analysis
"""

import json
from typing import Dict, List, Optional
from pathlib import Path


def create_ascii_bar_chart(data: Dict[str, float], title: str, max_width: int = 50) -> str:
    """
    Create ASCII bar chart for terminal display

    Args:
        data: Dict mapping labels to values
        title: Chart title
        max_width: Maximum width of bars in characters

    Returns:
        ASCII art bar chart as string
    """
    if not data:
        return f"{title}\n(No data)"

    # Find max value for scaling
    max_val = max(data.values())
    if max_val == 0:
        max_val = 1

    # Find max label length
    max_label_len = max(len(str(label)) for label in data.keys())

    # Build chart
    lines = [f"\n{title}", "=" * (max_width + max_label_len + 15)]

    for label, value in data.items():
        # Calculate bar length
        bar_len = int((value / max_val) * max_width)
        bar = "█" * bar_len

        # Format line
        label_padded = str(label).ljust(max_label_len)
        line = f"{label_padded} │ {bar} {value:.2f}"
        lines.append(line)

    lines.append("")
    return "\n".join(lines)


def create_comparison_table(measured: Dict, baseline: Dict, title: str) -> str:
    """
    Create ASCII comparison table

    Args:
        measured: Measured benchmark results
        baseline: Baseline values
        title: Table title

    Returns:
        ASCII table as string
    """
    lines = [f"\n{title}", "=" * 80]

    # Header
    header = f"{'Metric':<30} {'Measured':>15} {'Baseline':>15} {'% of Baseline':>15}"
    lines.append(header)
    lines.append("-" * 80)

    # Find common metrics
    common_metrics = set(measured.keys()) & set(baseline.keys())

    for metric in sorted(common_metrics):
        measured_val = measured[metric]
        baseline_val = baseline[metric]

        if baseline_val > 0:
            percent = (measured_val / baseline_val) * 100
        else:
            percent = 0

        row = f"{metric:<30} {measured_val:>15.2f} {baseline_val:>15.2f} {percent:>14.1f}%"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def create_performance_summary(results: Dict, gpu_name: str = "Unknown") -> str:
    """
    Create comprehensive performance summary with ASCII visualizations

    Args:
        results: Complete benchmark results dict
        gpu_name: GPU model name

    Returns:
        Formatted summary string
    """
    summary_lines = [
        "\n" + "=" * 80,
        f"PERFORMANCE SUMMARY - {gpu_name}",
        "=" * 80,
    ]

    # FLOPS performance
    if "flops" in results:
        flops_data = {}
        for precision, data in results["flops"].items():
            if isinstance(data, dict) and "median_gflops" in data:
                flops_data[precision.upper()] = data["median_gflops"] / 1000  # Convert to TFLOPS

        if flops_data:
            chart = create_ascii_bar_chart(flops_data, "FLOPS Performance (TFLOPS)", max_width=40)
            summary_lines.append(chart)

    # Memory bandwidth
    if "memory" in results and "bandwidth" in results["memory"]:
        bw_data = {}
        for transfer_type, data in results["memory"]["bandwidth"].items():
            if isinstance(data, dict) and "bandwidth_gbps" in data:
                bw_data[transfer_type.upper()] = data["bandwidth_gbps"]

        if bw_data:
            chart = create_ascii_bar_chart(bw_data, "Memory Bandwidth (GB/s)", max_width=40)
            summary_lines.append(chart)

    # Real-world workloads
    workload_data = {}

    if "resnet50" in results:
        if "inference" in results["resnet50"]:
            workload_data["ResNet50 Inference"] = results["resnet50"]["inference"].get("throughput_images_per_sec", 0)
        if "training_fp32" in results["resnet50"]:
            workload_data["ResNet50 Training"] = results["resnet50"]["training_fp32"].get("throughput_images_per_sec", 0)

    if "bert" in results:
        if "inference" in results["bert"]:
            workload_data["BERT Inference"] = results["bert"]["inference"].get("throughput_sequences_per_sec", 0)
        if "training_fp32" in results["bert"]:
            workload_data["BERT Training"] = results["bert"]["training_fp32"].get("throughput_sequences_per_sec", 0)

    if workload_data:
        chart = create_ascii_bar_chart(workload_data, "Real-World Workload Throughput", max_width=40)
        summary_lines.append(chart)

    # Power efficiency
    if "power" in results and "benchmarks" in results["power"]:
        power_data = {}
        for workload, data in results["power"]["benchmarks"].items():
            if isinstance(data, dict) and "avg_power_watts" in data:
                power_data[workload] = data["avg_power_watts"]

        if power_data:
            chart = create_ascii_bar_chart(power_data, "Power Consumption (Watts)", max_width=40)
            summary_lines.append(chart)

    # Thermal info
    if "thermal" in results:
        thermal_info = results["thermal"]
        summary_lines.extend([
            "\nThermal Information:",
            "-" * 80,
            f"  Average Temperature: {thermal_info.get('temperature', {}).get('avg_celsius', 'N/A')} °C",
            f"  Max Temperature: {thermal_info.get('temperature', {}).get('max_celsius', 'N/A')} °C",
            f"  Throttling Detected: {'Yes' if thermal_info.get('throttling_detected') else 'No'}",
        ])

    summary_lines.append("\n" + "=" * 80 + "\n")

    return "\n".join(summary_lines)


def export_html_report(results: Dict, output_path: str, gpu_name: str = "Unknown"):
    """
    Export results as HTML report (basic version without matplotlib)

    Args:
        results: Complete benchmark results
        output_path: Path to save HTML file
        gpu_name: GPU model name
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GPU Benchmark Report - {gpu_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ font-weight: bold; color: #4CAF50; font-size: 24px; }}
        .section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GPU Benchmark Report</h1>
        <p><strong>GPU:</strong> {gpu_name}</p>
        <p><strong>Date:</strong> {results.get('timestamp', 'N/A')}</p>

        <div class="section">
            <h2>FLOPS Performance</h2>
            <table>
                <tr><th>Precision</th><th>TFLOPS</th><th>Median (GFLOPS)</th></tr>
"""

    # Add FLOPS data
    if "flops" in results:
        for precision, data in results["flops"].items():
            if isinstance(data, dict):
                tflops = data.get("median_gflops", 0) / 1000
                gflops = data.get("median_gflops", 0)
                html += f"<tr><td>{precision.upper()}</td><td>{tflops:.2f}</td><td>{gflops:.2f}</td></tr>\n"

    html += """
            </table>
        </div>

        <div class="section">
            <h2>Memory Bandwidth</h2>
            <table>
                <tr><th>Transfer Type</th><th>Bandwidth (GB/s)</th></tr>
"""

    # Add memory bandwidth data
    if "memory" in results and "bandwidth" in results["memory"]:
        for transfer_type, data in results["memory"]["bandwidth"].items():
            if isinstance(data, dict):
                bw = data.get("bandwidth_gbps", 0)
                html += f"<tr><td>{transfer_type.upper()}</td><td>{bw:.2f}</td></tr>\n"

    html += """
            </table>
        </div>

        <div class="section">
            <h2>Real-World Workloads</h2>
            <table>
                <tr><th>Workload</th><th>Throughput</th><th>Unit</th></tr>
"""

    # Add workload data
    if "resnet50" in results:
        if "inference" in results["resnet50"]:
            throughput = results["resnet50"]["inference"].get("throughput_images_per_sec", 0)
            html += f"<tr><td>ResNet50 Inference</td><td>{throughput:.2f}</td><td>images/sec</td></tr>\n"
        if "training_fp32" in results["resnet50"]:
            throughput = results["resnet50"]["training_fp32"].get("throughput_images_per_sec", 0)
            html += f"<tr><td>ResNet50 Training (FP32)</td><td>{throughput:.2f}</td><td>images/sec</td></tr>\n"

    if "bert" in results:
        if "inference" in results["bert"]:
            throughput = results["bert"]["inference"].get("throughput_sequences_per_sec", 0)
            html += f"<tr><td>BERT Inference</td><td>{throughput:.2f}</td><td>sequences/sec</td></tr>\n"
        if "training_fp32" in results["bert"]:
            throughput = results["bert"]["training_fp32"].get("throughput_sequences_per_sec", 0)
            html += f"<tr><td>BERT Training (FP32)</td><td>{throughput:.2f}</td><td>sequences/sec</td></tr>\n"

    html += """
            </table>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")


if __name__ == "__main__":
    # Demo
    demo_data = {
        "FP32": 10.5,
        "FP16": 21.0,
        "TF32": 15.5,
        "BF16": 20.5,
    }

    print(create_ascii_bar_chart(demo_data, "FLOPS Performance (TFLOPS)"))
