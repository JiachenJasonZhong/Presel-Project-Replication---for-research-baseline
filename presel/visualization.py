"""
Visualization utilities for PreSel results.

Provides functions to visualize:
- Task distributions
- Selection results
- Method comparisons
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization functions will be disabled.")


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_task_distribution(
    task_sizes: Dict[str, int],
    output_file: str,
    title: str = "Task Distribution"
):
    """
    Plot distribution of samples across tasks.

    Args:
        task_sizes: Dictionary mapping task names to sizes
        output_file: Path to save figure
        title: Plot title
    """
    check_matplotlib()

    # Sort tasks by size
    sorted_tasks = sorted(task_sizes.items(), key=lambda x: x[1], reverse=True)
    tasks = [t[0] for t in sorted_tasks]
    sizes = [t[1] for t in sorted_tasks]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot
    bars = ax.bar(range(len(tasks)), sizes, color='steelblue', alpha=0.7)

    # Customize
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size)}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Task distribution plot saved to {output_file}")


def plot_task_weights(
    task_weights: Dict[str, float],
    task_sizes: Dict[str, int],
    output_file: str,
    title: str = "Task Importance Weights"
):
    """
    Plot task importance weights from PreSel.

    Args:
        task_weights: Dictionary mapping task names to weights
        task_sizes: Dictionary mapping task names to sizes
        output_file: Path to save figure
        title: Plot title
    """
    check_matplotlib()

    # Sort tasks by weight
    sorted_tasks = sorted(task_weights.items(), key=lambda x: x[1], reverse=True)
    tasks = [t[0] for t in sorted_tasks]
    weights = [t[1] for t in sorted_tasks]
    sizes = [task_sizes.get(t, 0) for t in tasks]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Task weights
    bars1 = ax1.bar(range(len(tasks)), weights, color='coral', alpha=0.7)
    ax1.set_xlabel('Task', fontsize=12)
    ax1.set_ylabel('Importance Weight w(T_i)', fontsize=12)
    ax1.set_title('Task Importance Weights', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(tasks)))
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, weight in zip(bars1, weights):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Weight vs Size comparison
    x = np.arange(len(tasks))
    width = 0.35

    # Normalize sizes to [0, 1] for comparison
    max_size = max(sizes) if sizes else 1
    normalized_sizes = [s / max_size for s in sizes]

    bars2a = ax2.bar(x - width/2, weights, width, label='Importance Weight', color='coral', alpha=0.7)
    bars2b = ax2.bar(x + width/2, normalized_sizes, width, label='Size (normalized)', color='steelblue', alpha=0.7)

    ax2.set_xlabel('Task', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Weight vs Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Task weights plot saved to {output_file}")


def plot_selection_comparison(
    results: Dict[str, Dict],
    output_file: str,
    title: str = "Method Comparison"
):
    """
    Compare selection results from different methods.

    Args:
        results: Dictionary mapping method names to results
            Each result should have 'task_counts' dict
        output_file: Path to save figure
        title: Plot title
    """
    check_matplotlib()

    # Get all unique tasks
    all_tasks = set()
    for result in results.values():
        all_tasks.update(result.get('task_counts', {}).keys())

    all_tasks = sorted(all_tasks)
    methods = list(results.keys())

    # Prepare data
    data = []
    for method in methods:
        task_counts = results[method].get('task_counts', {})
        counts = [task_counts.get(task, 0) for task in all_tasks]
        data.append(counts)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(all_tasks))
    width = 0.8 / len(methods)

    # Plot bars for each method
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for i, (method, counts) in enumerate(zip(methods, data)):
        offset = (i - len(methods)/2) * width + width/2
        bars = ax.bar(x + offset, counts, width, label=method, color=colors[i], alpha=0.8)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Number of Selected Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Selection comparison plot saved to {output_file}")


def plot_selection_summary(
    selected_samples: List[Dict],
    task_info: Dict,
    output_dir: str,
    prefix: str = "selection"
):
    """
    Create a comprehensive summary visualization.

    Args:
        selected_samples: List of selected samples
        task_info: Task information from PreSel
        output_dir: Directory to save plots
        prefix: Prefix for output files
    """
    check_matplotlib()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: Task distribution
    if 'task_sizes' in task_info:
        plot_task_distribution(
            task_sizes=task_info['task_sizes'],
            output_file=f"{output_dir}/{prefix}_task_distribution.png",
            title="Original Task Distribution"
        )

    # Plot 2: Task weights
    if 'task_weights' in task_info and 'task_sizes' in task_info:
        plot_task_weights(
            task_weights=task_info['task_weights'],
            task_sizes=task_info['task_sizes'],
            output_file=f"{output_dir}/{prefix}_task_weights.png",
            title="Task Importance Weights (PreSel)"
        )

    # Plot 3: Selection vs Original
    if 'task_sizes' in task_info:
        task_counts = {}
        for sample in selected_samples:
            task = sample.get('task')
            task_counts[task] = task_counts.get(task, 0) + 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original distribution
        sorted_tasks = sorted(task_info['task_sizes'].items(), key=lambda x: x[1], reverse=True)
        tasks = [t[0] for t in sorted_tasks]
        original_sizes = [t[1] for t in sorted_tasks]
        selected_sizes = [task_counts.get(t, 0) for t in tasks]

        # Plot original
        ax1.bar(range(len(tasks)), original_sizes, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Task', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Original Task Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(tasks)))
        ax1.set_xticklabels(tasks, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Plot selected
        ax2.bar(range(len(tasks)), selected_sizes, color='coral', alpha=0.7)
        ax2.set_xlabel('Task', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Selected Task Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(tasks)))
        ax2.set_xticklabels(tasks, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_before_after.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Before/after comparison saved to {output_dir}/{prefix}_before_after.png")

    print(f"\nAll visualizations saved to {output_dir}/")


def visualize_from_file(
    result_file: str,
    output_dir: str = None
):
    """
    Create visualizations from a saved result file.

    Args:
        result_file: Path to saved result JSON file
        output_dir: Directory to save plots (default: same as result file)
    """
    # Load results
    with open(result_file, 'r') as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = str(Path(result_file).parent / "visualizations")

    # Extract data
    selected_samples = data.get('selected_samples', [])
    task_info = data.get('task_info', {})

    # Create visualizations
    prefix = Path(result_file).stem
    plot_selection_summary(selected_samples, task_info, output_dir, prefix)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization.py <result_file.json> [output_dir]")
        sys.exit(1)

    result_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_from_file(result_file, output_dir)
