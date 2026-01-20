"""
Dataset loading utilities for VIT datasets.

Supports:
- LLaVA-1.5 format
- Vision-Flan format
- Custom VIT datasets
"""

import json
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


class VITDataset:
    """
    Visual Instruction Tuning Dataset wrapper.

    Organizes data by tasks for PreSel selection.
    """

    def __init__(self, tasks_data: Dict[str, Dict]):
        """
        Args:
            tasks_data: Dictionary mapping task names to task data.
                Each task should have:
                - 'images': List of image tensors or paths
                - 'questions': List of question texts (optional)
                - 'responses': List of response texts (optional)
                - 'image_ids': List of image IDs (optional)
        """
        self.tasks_data = tasks_data
        self.task_names = list(tasks_data.keys())

    def __len__(self):
        return sum(len(task_data['images']) for task_data in self.tasks_data.values())

    def get_task_sizes(self) -> Dict[str, int]:
        """Get number of images per task."""
        return {
            task_name: len(task_data['images'])
            for task_name, task_data in self.tasks_data.items()
        }

    def get_num_tasks(self) -> int:
        """Get number of tasks."""
        return len(self.task_names)

    def __getitem__(self, key):
        """Access task data by task name."""
        return self.tasks_data[key]

    def items(self):
        """Iterate over tasks."""
        return self.tasks_data.items()

    def keys(self):
        """Get task names."""
        return self.tasks_data.keys()

    def values(self):
        """Get task data."""
        return self.tasks_data.values()


def load_llava_dataset(
    annotation_file: str,
    image_dir: str,
    load_images: bool = True,
    max_samples_per_task: Optional[int] = None,
    tasks_to_load: Optional[List[str]] = None
) -> VITDataset:
    """
    Load LLaVA-1.5 format dataset.

    Expected annotation format (JSON):
    [
        {
            "id": "sample_id",
            "image": "image_filename.jpg",
            "conversations": [
                {"from": "human", "value": "question text"},
                {"from": "gpt", "value": "response text"}
            ],
            "task": "task_name"  # or inferred from source
        },
        ...
    ]

    Args:
        annotation_file: Path to annotation JSON file
        image_dir: Directory containing images
        load_images: Whether to load images into memory (vs storing paths)
        max_samples_per_task: Maximum samples per task (for debugging)
        tasks_to_load: List of specific tasks to load (None = all)

    Returns:
        VITDataset object
    """
    print(f"Loading LLaVA dataset from {annotation_file}")

    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations")

    # Organize by task
    tasks_data = {}

    for ann in tqdm(annotations, desc="Processing annotations"):
        # Get task name
        task_name = ann.get('task', 'unknown')

        # Infer task from source if not provided
        if task_name == 'unknown' and 'image' in ann:
            # Heuristic: use directory name or prefix
            if '/' in ann['image']:
                task_name = ann['image'].split('/')[0]
            else:
                task_name = 'llava'

        # Filter tasks if specified
        if tasks_to_load is not None and task_name not in tasks_to_load:
            continue

        # Initialize task if not exists
        if task_name not in tasks_data:
            tasks_data[task_name] = {
                'images': [],
                'questions': [],
                'responses': [],
                'image_ids': [],
                'metadata': []
            }

        # Check if task has reached max samples
        if max_samples_per_task is not None:
            if len(tasks_data[task_name]['images']) >= max_samples_per_task:
                continue

        # Get image path
        image_filename = ann.get('image', '')
        image_path = os.path.join(image_dir, image_filename)

        # Load or store image
        if load_images and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                tasks_data[task_name]['images'].append(img_tensor)
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                continue
        else:
            # Store path instead
            tasks_data[task_name]['images'].append(image_path)

        # Extract question and response from conversations
        conversations = ann.get('conversations', [])
        question = ""
        response = ""

        for conv in conversations:
            if conv.get('from') == 'human':
                question += conv.get('value', '') + " "
            elif conv.get('from') == 'gpt':
                response += conv.get('value', '') + " "

        tasks_data[task_name]['questions'].append(question.strip())
        tasks_data[task_name]['responses'].append(response.strip())
        tasks_data[task_name]['image_ids'].append(ann.get('id', ''))
        tasks_data[task_name]['metadata'].append(ann)

    # Print statistics
    print(f"\nDataset loaded with {len(tasks_data)} tasks:")
    for task_name, task_data in sorted(tasks_data.items(), key=lambda x: len(x[1]['images']), reverse=True):
        print(f"  {task_name}: {len(task_data['images'])} samples")

    return VITDataset(tasks_data)


def load_vision_flan_dataset(
    annotation_file: str,
    image_dir: str,
    load_images: bool = True,
    max_samples_per_task: Optional[int] = None,
    tasks_to_load: Optional[List[str]] = None
) -> VITDataset:
    """
    Load Vision-Flan format dataset.

    Expected annotation format similar to LLaVA but with more tasks.

    Args:
        annotation_file: Path to annotation JSON file
        image_dir: Directory containing images
        load_images: Whether to load images into memory
        max_samples_per_task: Maximum samples per task
        tasks_to_load: List of specific tasks to load

    Returns:
        VITDataset object
    """
    # Vision-Flan uses similar format to LLaVA
    return load_llava_dataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        load_images=load_images,
        max_samples_per_task=max_samples_per_task,
        tasks_to_load=tasks_to_load
    )


def create_dummy_dataset(
    num_tasks: int = 5,
    samples_per_task: int = 100,
    image_size: tuple = (224, 224)
) -> VITDataset:
    """
    Create a dummy dataset for testing.

    Args:
        num_tasks: Number of tasks
        samples_per_task: Number of samples per task
        image_size: Image dimensions (H, W)

    Returns:
        VITDataset object
    """
    print(f"Creating dummy dataset: {num_tasks} tasks, {samples_per_task} samples each")

    tasks_data = {}

    for i in range(num_tasks):
        task_name = f"task_{i}"

        # Create random images
        images = []
        questions = []
        responses = []

        for j in range(samples_per_task):
            # Random image
            img = torch.rand(3, image_size[0], image_size[1])
            images.append(img)

            # Dummy question and response
            questions.append(f"What is in this image from {task_name}?")
            responses.append(f"This is sample {j} from {task_name}.")

        tasks_data[task_name] = {
            'images': images,
            'questions': questions,
            'responses': responses,
            'image_ids': [f"{task_name}_{j}" for j in range(samples_per_task)]
        }

    print(f"Created dummy dataset with {len(tasks_data)} tasks")

    return VITDataset(tasks_data)