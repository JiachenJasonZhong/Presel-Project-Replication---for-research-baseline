#!/usr/bin/env python
"""
PreSel 数据选择脚本

用法:
  # dummy 数据测试
  python run_presel.py --data_type dummy

  # 真实数据
  python run_presel.py --data_type llava --annotation_file data.json --image_dir ./images
"""

import os
import argparse

from presel import PreSel
from presel.presel import PreSelBaseline
from presel.data import load_llava_dataset, create_dummy_dataset, save_selection_results


def main():
    parser = argparse.ArgumentParser()

    # 数据参数
    parser.add_argument("--data_type", default="dummy", choices=["llava", "vision_flan", "dummy"])
    parser.add_argument("--annotation_file", type=str, help="annotation json 路径")
    parser.add_argument("--image_dir", type=str, help="图片目录")

    # dummy 数据参数
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--samples_per_task", type=int, default=500)

    # 选择参数
    parser.add_argument("--sampling_ratio", type=float, default=0.15, help="选择比例，默认 15%%")
    parser.add_argument("--reference_ratio", type=float, default=0.05)
    parser.add_argument("--method", default="presel", choices=["presel", "random", "uniform", "size_balanced"])

    # 输出
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--exp_name", default="presel_run")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Method: {args.method}, Sampling ratio: {args.sampling_ratio}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    if args.data_type == "dummy":
        dataset = create_dummy_dataset(args.num_tasks, args.samples_per_task)
    else:
        if not args.annotation_file or not args.image_dir:
            raise ValueError("需要指定 --annotation_file 和 --image_dir")
        dataset = load_llava_dataset(args.annotation_file, args.image_dir)

    print(f"Dataset: {len(dataset)} samples, {dataset.get_num_tasks()} tasks")

    # 运行选择
    if args.method == "presel":
        selector = PreSel(
            sampling_ratio=args.sampling_ratio,
            reference_ratio=args.reference_ratio,
            use_simplified=True,
            seed=args.seed
        )
    else:
        selector = PreSelBaseline(args.method, args.sampling_ratio, args.seed)

    selected, task_info = selector.select(dataset)

    # 保存结果
    output_file = os.path.join(args.output_dir, f"{args.exp_name}_{args.method}.json")
    save_selection_results(selected, task_info, output_file)

    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
