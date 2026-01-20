"""
PreSel: Pre-Instruction Data Selection for Visual Instruction Tuning

Main pipeline that combines task-importance estimation and cluster-based selection.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .irs_calculator import IRSCalculator, SimplifiedIRSCalculator
from .task_importance import TaskImportanceEstimator, UniformTaskWeighting, SizeBalancedWeighting
from .cluster_selector import ClusterBasedSelector, RandomSelector


class PreSel:
    """
    PreSel: Pre-Instruction Data Selection for Visual Instruction Tuning

    Main pipeline that performs:
    1. Task-importance estimation using a small reference set
    2. Cluster-based selection of representative images
    """

    def __init__(
        self,
        reference_model=None,
        tokenizer=None,
        feature_extractor=None,
        sampling_ratio: float = 0.15,
        reference_ratio: float = 0.05,
        num_neighbors: int = 5,
        temperature: float = None,
        use_simplified: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ):
        """
        Args:
            reference_model: Pre-trained LVLM for IRS computation (optional)
            tokenizer: Tokenizer for the reference model (optional)
            feature_extractor: Vision encoder (e.g., DINOv2) for feature extraction (optional)
            sampling_ratio: Proportion of data to select (default: 0.15 = 15%)
            reference_ratio: Proportion of data for reference set (default: 0.05 = 5%)
            num_neighbors: k in k-nearest neighbors for NC score (default: 5)
            temperature: Temperature for task-importance softmax (default: sqrt(1/M))
            use_simplified: Whether to use simplified IRS calculator (default: True)
            device: Device to run computations on
            seed: Random seed
        """
        self.sampling_ratio = sampling_ratio
        self.reference_ratio = reference_ratio
        self.num_neighbors = num_neighbors
        self.device = device
        self.seed = seed

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize IRS calculator
        if use_simplified or reference_model is None:
            irs_calculator = SimplifiedIRSCalculator(device=device)
        else:
            irs_calculator = IRSCalculator(
                model=reference_model,
                tokenizer=tokenizer,
                device=device
            )

        # Initialize task-importance estimator
        self.task_estimator = TaskImportanceEstimator(
            irs_calculator=irs_calculator,
            temperature=temperature,
            use_simplified=use_simplified
        )

        # Initialize cluster-based selector
        self.cluster_selector = ClusterBasedSelector(
            feature_extractor=feature_extractor,
            num_neighbors=num_neighbors,
            device=device
        )

        # Storage for results
        self.task_weights = {}
        self.selected_indices = {}

    def select(
        self,
        dataset: Dict[str, Dict],
        return_task_info: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Main selection pipeline.

        Args:
            dataset: Dictionary mapping task names to task data
                Each task should have:
                - 'images': List of image tensors
                - 'questions': List of question texts (optional, for reference set)
                - 'responses': List of response texts (optional, for reference set)
            return_task_info: Whether to return detailed task information

        Returns:
            Tuple of (selected_samples, task_info)
            - selected_samples: List of selected image data
            - task_info: Dictionary with task weights and budgets
        """
        print("=" * 80)
        print("PreSel: Pre-Instruction Data Selection")
        print("=" * 80)

        # Step 1: Create reference set for task-importance estimation
        print("\n[Step 1] Creating reference set...")
        reference_samples = self._create_reference_set(dataset)

        print(f"Reference set size: {len(reference_samples)} samples")
        print(f"Tasks in reference set: {len(set(s['task'] for s in reference_samples))}")

        # Step 2: Estimate task importance
        print("\n[Step 2] Estimating task importance...")
        self.task_weights = self.task_estimator.estimate_task_importance(
            reference_samples
        )

        # Step 3: Compute task budgets for selection
        print("\n[Step 3] Computing task budgets...")
        total_images = sum(len(task_data['images']) for task_data in dataset.values())
        # Total budget = sampling_ratio * total_images
        # But we already used reference_ratio for reference set
        # So additional selection budget = (sampling_ratio - reference_ratio) * total_images
        additional_budget = int((self.sampling_ratio - self.reference_ratio) * total_images)

        task_sizes = {
            task_name: len(task_data['images'])
            for task_name, task_data in dataset.items()
        }

        task_budgets = self.task_estimator.get_all_task_budgets(
            total_budget=additional_budget,
            task_sizes=task_sizes
        )

        print(f"\nTotal images: {total_images}")
        print(f"Total budget: {int(self.sampling_ratio * total_images)} ({self.sampling_ratio*100:.1f}%)")
        print(f"Reference set: {len(reference_samples)}")
        print(f"Additional selection: {additional_budget}")

        print("\nTask budgets:")
        for task_name, budget in sorted(task_budgets.items(), key=lambda x: x[1], reverse=True):
            task_size = task_sizes[task_name]
            percentage = (budget / task_size * 100) if task_size > 0 else 0
            print(f"  {task_name}: {budget}/{task_size} ({percentage:.1f}%)")

        # Step 4: Select images from each task
        print("\n[Step 4] Selecting representative images from each task...")
        all_selected = []

        # First, add reference samples to selection
        reference_indices_by_task = defaultdict(list)
        for i, sample in enumerate(reference_samples):
            task = sample['task']
            reference_indices_by_task[task].append(sample['original_index'])

        for task_name, task_data in dataset.items():
            images = task_data['images']
            budget = task_budgets.get(task_name, 0)

            if budget <= 0:
                continue

            # Get indices already selected in reference set
            ref_indices = set(reference_indices_by_task.get(task_name, []))

            # Get remaining images for selection
            remaining_indices = [i for i in range(len(images)) if i not in ref_indices]
            remaining_images = [images[i] for i in remaining_indices]

            if len(remaining_images) == 0:
                continue

            # Select from remaining images
            selected_local_indices = self.cluster_selector.select_images(
                images=remaining_images,
                budget=budget,
                task_name=task_name,
                show_progress=True
            )

            # Map local indices back to original indices
            selected_indices = [remaining_indices[i] for i in selected_local_indices]

            # Store selected indices for this task
            # Combine reference indices and newly selected indices
            all_task_indices = list(ref_indices) + selected_indices
            self.selected_indices[task_name] = all_task_indices

            # Create selected samples
            for idx in all_task_indices:
                sample = {
                    'task': task_name,
                    'image': images[idx],
                    'original_index': idx
                }
                # Add additional fields if available
                for key in ['question', 'response', 'image_id', 'metadata']:
                    if key in task_data and idx < len(task_data[key]):
                        sample[key] = task_data[key][idx]

                all_selected.append(sample)

        # Summary
        print("\n" + "=" * 80)
        print("Selection Summary")
        print("=" * 80)
        print(f"Total images in dataset: {total_images}")
        print(f"Total selected: {len(all_selected)} ({len(all_selected)/total_images*100:.1f}%)")
        print(f"Target ratio: {self.sampling_ratio*100:.1f}%")

        # Prepare task info
        task_info = {
            'task_weights': self.task_weights,
            'task_budgets': task_budgets,
            'task_sizes': task_sizes,
            'selected_indices': self.selected_indices,
            'reference_samples': reference_samples
        }

        if return_task_info:
            return all_selected, task_info
        else:
            return all_selected

    def _create_reference_set(
        self,
        dataset: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Create a small reference set for task-importance estimation.

        Randomly samples reference_ratio of data from each task.

        Args:
            dataset: Dictionary mapping task names to task data

        Returns:
            List of reference samples with questions and responses
        """
        reference_samples = []

        for task_name, task_data in dataset.items():
            images = task_data['images']
            num_images = len(images)

            # Calculate number of reference samples for this task
            num_reference = max(1, int(num_images * self.reference_ratio))

            # Randomly sample indices
            reference_indices = np.random.choice(
                num_images,
                size=min(num_reference, num_images),
                replace=False
            )

            # Create reference samples
            for idx in reference_indices:
                sample = {
                    'task': task_name,
                    'image': images[idx],
                    'original_index': int(idx)
                }

                # Add question and response if available
                if 'questions' in task_data and idx < len(task_data['questions']):
                    sample['question'] = task_data['questions'][idx]
                else:
                    # Use dummy question if not provided
                    sample['question'] = f"What can you see in this image from {task_name}?"

                if 'responses' in task_data and idx < len(task_data['responses']):
                    sample['response'] = task_data['responses'][idx]
                else:
                    # Use dummy response if not provided
                    sample['response'] = f"This is an image from the {task_name} task."

                reference_samples.append(sample)

        return reference_samples


class PreSelBaseline:
    """
    Baseline methods for comparison.
    """

    def __init__(
        self,
        method: str = "random",
        sampling_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            method: Baseline method ('random', 'uniform', 'size_balanced')
            sampling_ratio: Proportion of data to select
            seed: Random seed
        """
        self.method = method
        self.sampling_ratio = sampling_ratio
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def select(
        self,
        dataset: Dict[str, Dict],
        return_task_info: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """Select images using baseline method."""
        total_images = sum(len(task_data['images']) for task_data in dataset.values())
        total_budget = int(self.sampling_ratio * total_images)

        print(f"\nBaseline Method: {self.method}")
        print(f"Total images: {total_images}")
        print(f"Total budget: {total_budget} ({self.sampling_ratio*100:.1f}%)")

        all_selected = []
        selected_indices = {}

        if self.method == "random":
            # Random selection across all tasks
            all_images = []
            image_to_task = []

            for task_name, task_data in dataset.items():
                all_images.extend(task_data['images'])
                image_to_task.extend([task_name] * len(task_data['images']))

            selected_idx = np.random.choice(
                len(all_images),
                size=total_budget,
                replace=False
            )

            for idx in selected_idx:
                task_name = image_to_task[idx]
                sample = {
                    'task': task_name,
                    'image': all_images[idx],
                    'original_index': idx
                }
                all_selected.append(sample)

        elif self.method == "uniform":
            # Uniform selection: equal budget per task
            num_tasks = dataset.get_num_tasks()
            budget_per_task = total_budget // num_tasks

            for task_name, task_data in dataset.items():
                images = task_data['images']
                budget = min(budget_per_task, len(images))

                selected_idx = np.random.choice(
                    len(images),
                    size=budget,
                    replace=False
                )
                selected_indices[task_name] = selected_idx.tolist()

                for idx in selected_idx:
                    sample = {
                        'task': task_name,
                        'image': images[idx],
                        'original_index': int(idx)
                    }
                    all_selected.append(sample)

        elif self.method == "size_balanced":
            # Size-balanced: budget proportional to task size
            task_sizes = {
                task_name: len(task_data['images'])
                for task_name, task_data in dataset.items()
            }

            for task_name, task_data in dataset.items():
                images = task_data['images']
                task_budget = int(total_budget * len(images) / total_images)
                task_budget = min(task_budget, len(images))

                selected_idx = np.random.choice(
                    len(images),
                    size=task_budget,
                    replace=False
                )
                selected_indices[task_name] = selected_idx.tolist()

                for idx in selected_idx:
                    sample = {
                        'task': task_name,
                        'image': images[idx],
                        'original_index': int(idx)
                    }
                    all_selected.append(sample)

        print(f"Selected {len(all_selected)} images")

        task_info = {
            'selected_indices': selected_indices,
            'method': self.method
        }

        if return_task_info:
            return all_selected, task_info
        else:
            return all_selected
