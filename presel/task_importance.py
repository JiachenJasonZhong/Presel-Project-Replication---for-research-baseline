"""
Task-Importance Estimation Module

Implements the task-importance estimation mechanism described in Section 3.1:
1. Compute IRS for samples in reference set
2. Average IRS per task to get s(T_i)
3. Derive task weights w(T_i) using softmax with temperature
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict
from .irs_calculator import IRSCalculator, SimplifiedIRSCalculator


class TaskImportanceEstimator:
    """
    Estimate the relative importance of each vision task for VIT.

    Uses Instruction Relevance Score (IRS) to determine how much budget
    should be allocated to each task in the final selection.
    """

    def __init__(
        self,
        irs_calculator: IRSCalculator = None,
        temperature: float = None,
        use_simplified: bool = False
    ):
        """
        Args:
            irs_calculator: IRSCalculator instance (or None to create SimplifiedIRSCalculator)
            temperature: Temperature for softmax (default: sqrt(1/M) where M is num tasks)
            use_simplified: Whether to use simplified IRS calculator
        """
        if irs_calculator is None or use_simplified:
            self.irs_calculator = SimplifiedIRSCalculator()
        else:
            self.irs_calculator = irs_calculator

        self.temperature = temperature
        self.task_scores = {}
        self.task_weights = {}

    def estimate_task_importance(
        self,
        reference_samples: List[Dict],
        task_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Estimate importance weights for each task.

        Args:
            reference_samples: List of reference samples with keys:
                - 'image': image tensor
                - 'question': question text
                - 'response': response text
                - 'task': task name/ID
            task_names: Optional list of task names (inferred if not provided)

        Returns:
            Dictionary mapping task names to importance weights w(T_i)
        """
        # Step 1: Compute IRS for all reference samples
        print("Computing IRS for reference samples...")
        irs_scores = self.irs_calculator.compute_batch_irs(
            reference_samples,
            show_progress=True
        )

        # Step 2: Group IRS by task and compute average s(T_i)
        print("Computing task-level importance scores...")
        task_irs = defaultdict(list)

        for sample, irs in zip(reference_samples, irs_scores):
            task = sample.get('task', 'unknown')
            task_irs[task].append(irs)

        # Compute average IRS per task (Equation 4)
        self.task_scores = {}
        for task, irs_list in task_irs.items():
            self.task_scores[task] = np.mean(irs_list)

        # Step 3: Compute task weights w(T_i) using softmax (Equation 5)
        self.task_weights = self._compute_task_weights(self.task_scores)

        print("\nTask Importance Scores:")
        for task in sorted(self.task_weights.keys()):
            print(f"  {task}: s(T_i)={self.task_scores[task]:.4f}, "
                  f"w(T_i)={self.task_weights[task]:.4f}")

        return self.task_weights

    def _compute_task_weights(self, task_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute task weights using softmax with temperature (Equation 5).

        w(T_i) = exp(-s(T_i) / τ) / Σ_j exp(-s(T_j) / τ)

        Args:
            task_scores: Dictionary of task scores s(T_i)

        Returns:
            Dictionary of task weights w(T_i)
        """
        tasks = list(task_scores.keys())
        scores = np.array([task_scores[t] for t in tasks])

        # Set temperature as sqrt(1/M) if not provided
        if self.temperature is None:
            M = len(tasks)
            self.temperature = np.sqrt(1.0 / M)

        # Compute softmax with temperature (note the negative sign)
        # Lower IRS means higher importance, so we use -s(T_i)
        exp_scores = np.exp(-scores / self.temperature)
        weights = exp_scores / np.sum(exp_scores)

        # Create dictionary mapping tasks to weights
        task_weights = {task: float(weight) for task, weight in zip(tasks, weights)}

        return task_weights

    def get_task_budget(
        self,
        task_name: str,
        total_budget: int,
        task_size: int = None
    ) -> int:
        """
        Get the sampling budget for a specific task.

        Args:
            task_name: Name of the task
            total_budget: Total number of samples to select
            task_size: Size of the task (optional, for bounds checking)

        Returns:
            Number of samples to select from this task
        """
        if task_name not in self.task_weights:
            raise ValueError(f"Task '{task_name}' not found in estimated weights")

        weight = self.task_weights[task_name]
        budget = int(weight * total_budget)

        # Ensure budget doesn't exceed task size
        if task_size is not None:
            budget = min(budget, task_size)

        return budget

    def get_all_task_budgets(
        self,
        total_budget: int,
        task_sizes: Dict[str, int] = None
    ) -> Dict[str, int]:
        """
        Get sampling budgets for all tasks.

        Args:
            total_budget: Total number of samples to select
            task_sizes: Dictionary mapping task names to their sizes

        Returns:
            Dictionary mapping task names to their budgets
        """
        budgets = {}

        for task_name, weight in self.task_weights.items():
            task_size = task_sizes.get(task_name) if task_sizes else None
            budgets[task_name] = self.get_task_budget(
                task_name,
                total_budget,
                task_size
            )

        # Adjust budgets to match total_budget exactly
        budgets = self._adjust_budgets(budgets, total_budget)

        return budgets

    def _adjust_budgets(
        self,
        budgets: Dict[str, int],
        total_budget: int
    ) -> Dict[str, int]:
        """
        Adjust budgets to match total_budget exactly.

        Due to rounding, the sum of individual budgets may not equal
        the total budget. This method adjusts by adding/removing samples
        from tasks proportionally.
        """
        current_total = sum(budgets.values())
        difference = total_budget - current_total

        if difference == 0:
            return budgets

        # Sort tasks by their weights (descending)
        sorted_tasks = sorted(
            budgets.keys(),
            key=lambda t: self.task_weights[t],
            reverse=True
        )

        # Distribute the difference
        adjusted_budgets = budgets.copy()

        if difference > 0:
            # Add samples to top tasks
            for i in range(difference):
                task = sorted_tasks[i % len(sorted_tasks)]
                adjusted_budgets[task] += 1
        else:
            # Remove samples from bottom tasks
            for i in range(abs(difference)):
                task = sorted_tasks[-(i % len(sorted_tasks)) - 1]
                if adjusted_budgets[task] > 0:
                    adjusted_budgets[task] -= 1

        return adjusted_budgets


class UniformTaskWeighting:
    """
    Baseline: Uniform task weighting (all tasks get equal budget).
    """

    def __init__(self):
        self.task_weights = {}

    def estimate_task_importance(
        self,
        reference_samples: List[Dict],
        task_names: List[str] = None
    ) -> Dict[str, float]:
        """Assign uniform weights to all tasks."""
        if task_names is None:
            task_names = list(set(s.get('task', 'unknown') for s in reference_samples))

        num_tasks = len(task_names)
        weight = 1.0 / num_tasks

        self.task_weights = {task: weight for task in task_names}
        return self.task_weights

    def get_all_task_budgets(
        self,
        total_budget: int,
        task_sizes: Dict[str, int] = None
    ) -> Dict[str, int]:
        """Get uniform budgets for all tasks."""
        num_tasks = len(self.task_weights)
        base_budget = total_budget // num_tasks
        remainder = total_budget % num_tasks

        budgets = {}
        for i, task in enumerate(self.task_weights.keys()):
            budgets[task] = base_budget + (1 if i < remainder else 0)

        return budgets


class SizeBalancedWeighting:
    """
    Baseline: Size-balanced weighting (budget proportional to task size).
    """

    def __init__(self):
        self.task_weights = {}
        self.task_sizes = {}

    def estimate_task_importance(
        self,
        reference_samples: List[Dict],
        task_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Assign weights proportional to task sizes in reference set.
        """
        task_counts = defaultdict(int)
        for sample in reference_samples:
            task = sample.get('task', 'unknown')
            task_counts[task] += 1

        total_samples = len(reference_samples)
        self.task_sizes = dict(task_counts)
        self.task_weights = {
            task: count / total_samples
            for task, count in task_counts.items()
        }

        return self.task_weights

    def get_all_task_budgets(
        self,
        total_budget: int,
        task_sizes: Dict[str, int] = None
    ) -> Dict[str, int]:
        """Get size-balanced budgets."""
        if task_sizes is not None:
            # Use provided task sizes
            total_size = sum(task_sizes.values())
            budgets = {
                task: int(size / total_size * total_budget)
                for task, size in task_sizes.items()
            }
        else:
            # Use weights from reference set
            budgets = {
                task: int(weight * total_budget)
                for task, weight in self.task_weights.items()
            }

        # Adjust to match total budget
        current_total = sum(budgets.values())
        if current_total < total_budget:
            # Add remaining to largest task
            largest_task = max(budgets.keys(), key=lambda t: budgets[t])
            budgets[largest_task] += total_budget - current_total

        return budgets