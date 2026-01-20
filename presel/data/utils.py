"""
Utility functions for data handling.
"""

import json
import os
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_images_from_paths(
    image_paths: List[str],
    show_progress: bool = True
) -> List[torch.Tensor]:
    """
    Load images from file paths.

    Args:
        image_paths: List of image file paths
        show_progress: Whether to show progress bar

    Returns:
        List of image tensors
    """
    images = []

    iterator = tqdm(image_paths, desc="Loading images") if show_progress else image_paths

    for path in iterator:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            # Use blank image as fallback
            images.append(torch.zeros(3, 224, 224))

    return images


def save_selection_results(
    selected_samples: List[Dict],
    task_info: Dict,
    output_file: str,
    format: str = "json"
):
    """
    Save selection results to file.

    Args:
        selected_samples: List of selected samples
        task_info: Task information from PreSel
        output_file: Output file path
        format: Format to save ("json" or "pickle")
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    save_data = {
        'selected_samples': [],
        'task_info': {
            'task_weights': task_info.get('task_weights', {}),
            'task_budgets': task_info.get('task_budgets', {}),
            'task_sizes': task_info.get('task_sizes', {}),
        },
        'statistics': {
            'total_selected': len(selected_samples),
            'num_tasks': len(task_info.get('task_weights', {}))
        }
    }

    # Convert samples for saving (remove image tensors)
    for sample in selected_samples:
        # Convert numpy types to Python types
        original_idx = sample.get('original_index')
        if isinstance(original_idx, np.integer):
            original_idx = int(original_idx)

        sample_data = {
            'task': sample.get('task'),
            'original_index': original_idx,
            'image_id': sample.get('image_id', ''),
        }

        # Add optional fields
        for key in ['question', 'response', 'metadata']:
            if key in sample:
                sample_data[key] = sample[key]

        save_data['selected_samples'].append(sample_data)

    # Save indices by task for easy access (convert numpy types to Python types)
    selected_indices = task_info.get('selected_indices', {})
    save_data['selected_indices_by_task'] = {
        task: [int(idx) for idx in indices] if isinstance(indices, (list, np.ndarray)) else indices
        for task, indices in selected_indices.items()
    }

    # Save to file
    if format == "json":
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {output_file} (JSON)")

    elif format == "pickle":
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Results saved to {output_file} (Pickle)")

    else:
        raise ValueError(f"Unsupported format: {format}")

    # Also save a summary text file
    summary_file = output_path.with_suffix('.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PreSel Selection Results Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total selected: {len(selected_samples)}\n")
        f.write(f"Number of tasks: {len(task_info.get('task_weights', {}))}\n\n")

        f.write("Task Weights:\n")
        for task, weight in sorted(
            task_info.get('task_weights', {}).items(),
            key=lambda x: x[1],
            reverse=True
        ):
            budget = task_info.get('task_budgets', {}).get(task, 0)
            size = task_info.get('task_sizes', {}).get(task, 0)
            f.write(f"  {task}: w={weight:.4f}, budget={budget}/{size}\n")

        f.write("\nSelected samples by task:\n")
        task_counts = {}
        for sample in selected_samples:
            task = sample.get('task')
            task_counts[task] = task_counts.get(task, 0) + 1

        for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {task}: {count} samples\n")

    print(f"Summary saved to {summary_file}")


def load_selection_results(
    input_file: str,
    format: str = "auto"
) -> Dict:
    """
    Load selection results from file.

    Args:
        input_file: Input file path
        format: Format to load ("json", "pickle", or "auto")

    Returns:
        Dictionary with selection results
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Auto-detect format
    if format == "auto":
        if input_path.suffix == '.json':
            format = "json"
        elif input_path.suffix in ['.pkl', '.pickle']:
            format = "pickle"
        else:
            raise ValueError(f"Cannot auto-detect format for {input_file}")

    # Load from file
    if format == "json":
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"Results loaded from {input_file} (JSON)")

    elif format == "pickle":
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Results loaded from {input_file} (Pickle)")

    else:
        raise ValueError(f"Unsupported format: {format}")

    return data


def export_selected_indices(
    task_info: Dict,
    output_file: str
):
    """
    Export only the selected indices to a simple JSON file.

    This is useful for downstream tasks that just need to know which
    samples were selected.

    Args:
        task_info: Task information from PreSel
        output_file: Output JSON file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices_data = {
        'selected_indices_by_task': task_info.get('selected_indices', {}),
        'task_budgets': task_info.get('task_budgets', {}),
        'task_weights': task_info.get('task_weights', {})
    }

    with open(output_file, 'w') as f:
        json.dump(indices_data, f, indent=2)

    print(f"Selected indices exported to {output_file}")


def load_json_annotations(file_path: str) -> List[Dict]:
    """
    Load annotations from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of annotation dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Some datasets store as dict with 'annotations' key
        return data.get('annotations', [data])
    else:
        raise ValueError(f"Unexpected JSON structure in {file_path}")
