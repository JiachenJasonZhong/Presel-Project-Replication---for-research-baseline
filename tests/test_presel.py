"""测试 PreSel 基本功能"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from presel import PreSel
from presel.data import create_dummy_dataset


def test_basic():
    print("Creating dummy dataset...")
    dataset = create_dummy_dataset(num_tasks=3, samples_per_task=50)
    print(f"Dataset: {len(dataset)} samples, {dataset.get_num_tasks()} tasks")

    print("\nRunning PreSel...")
    selector = PreSel(
        sampling_ratio=0.15,
        reference_ratio=0.05,
        num_neighbors=5,
        use_simplified=True
    )

    selected, task_info = selector.select(dataset)

    print(f"\nResult: {len(selected)} samples selected ({len(selected)/len(dataset)*100:.1f}%)")
    print("\nTask weights:")
    for task, weight in sorted(task_info['task_weights'].items(), key=lambda x: x[1], reverse=True):
        budget = task_info['task_budgets'][task]
        print(f"  {task}: weight={weight:.4f}, budget={budget}")

    return True


def test_baselines():
    """测试 baseline 方法"""
    from presel.presel import PreSelBaseline

    dataset = create_dummy_dataset(num_tasks=3, samples_per_task=100)

    for method in ['random', 'uniform', 'size_balanced']:
        selector = PreSelBaseline(method=method, sampling_ratio=0.15)
        selected, _ = selector.select(dataset)
        print(f"{method}: {len(selected)} selected")


if __name__ == "__main__":
    print("=" * 60)
    print("PreSel Test")
    print("=" * 60)

    test_basic()

    print("\n" + "=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    test_baselines()

    print("\nTest passed.")