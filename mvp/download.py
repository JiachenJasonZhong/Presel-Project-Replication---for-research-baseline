"""下载验证所需的模型和数据"""

import os

def download_dinov2():
    """下载 DINOv2 模型"""
    print("Downloading DINOv2...")
    import torch
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    print("DINOv2 downloaded.")
    return model


def download_sample_data(output_dir="./data", num_samples=100):
    """
    从 HuggingFace 下载 Vision-Flan 小样本
    或者生成模拟真实结构的测试数据
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading sample VIT data...")

    try:
        from datasets import load_dataset
        # 尝试加载 Vision-Flan 的一小部分
        ds = load_dataset("Vision-Flan/vision-flan_191-task_1k", split="train", streaming=True)
        samples = list(ds.take(num_samples))
        print(f"Downloaded {len(samples)} samples from Vision-Flan")
        return samples
    except Exception as e:
        print(f"Failed to download Vision-Flan: {e}")
        print("Using LLaVA-Instruct-150K instead...")

        try:
            ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
            samples = list(ds.take(num_samples))
            print(f"Downloaded {len(samples)} samples")
            return samples
        except Exception as e2:
            print(f"Failed: {e2}")
            print("Please install: pip install datasets")
            return None


if __name__ == "__main__":
    print("=" * 60)
    print("MVP: Download Models and Data")
    print("=" * 60)

    # 1. DINOv2
    print("\n[1/2] DINOv2")
    try:
        model = download_dinov2()
        print("Success")
    except Exception as e:
        print(f"Failed: {e}")

    # 2. Sample data
    print("\n[2/2] Sample Data")
    samples = download_sample_data(num_samples=100)
    if samples:
        print("Success")
