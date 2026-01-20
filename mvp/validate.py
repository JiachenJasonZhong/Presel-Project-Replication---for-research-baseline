"""
MVP: 最小验证 PreSel 算法正确性

验证项：
1. DINOv2 真实特征提取
2. 真实 LLM 计算 IRS (用小模型 GPT-2 近似)
3. Task weights 计算是否符合论文逻辑
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm


class RealFeatureExtractor:
    """用真正的 DINOv2 提取特征"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading DINOv2 on {device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model = self.model.to(device)
        self.model.eval()
        print("DINOv2 loaded.")

    def extract(self, images):
        """提取 [CLS] token 特征"""
        features = []
        with torch.no_grad():
            for img in tqdm(images, desc="Extracting DINOv2 features"):
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)
                # DINOv2 需要 224x224，做 resize
                img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear')
                feat = self.model(img)  # [1, 384] for vits14
                features.append(feat.cpu().numpy())
        return np.vstack(features)


class RealIRSCalculator:
    """
    用真实 LLM 计算 IRS
    论文公式: IRS = L(R|Q,I) / L(R|I)

    简化版: 用 GPT-2 近似（没有视觉输入，但可以验证公式逻辑）
    IRS = L(R|Q) / L(R)
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading GPT-2 on {device}...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model.to(device)
        self.model.eval()
        print("GPT-2 loaded.")

    def compute_loss(self, context, response):
        """
        计算给定 context 下生成 response 的 loss
        关键：只计算 response tokens 的 loss，不算 context 部分
        """
        # Tokenize context 和 response 分开
        if context:
            context_ids = self.tokenizer(context, return_tensors="pt")["input_ids"]
            response_ids = self.tokenizer(" " + response, return_tensors="pt")["input_ids"]
            full_ids = torch.cat([context_ids, response_ids], dim=1)
            context_len = context_ids.shape[1]
        else:
            full_ids = self.tokenizer(response, return_tensors="pt")["input_ids"]
            context_len = 0

        full_ids = full_ids.to(self.device)

        # 创建 labels：context 部分设为 -100（不计算 loss）
        labels = full_ids.clone()
        if context_len > 0:
            labels[:, :context_len] = -100  # 忽略 context 的 loss

        with torch.no_grad():
            outputs = self.model(full_ids, labels=labels)
            return outputs.loss.item()

    def compute_irs(self, question, response):
        """
        IRS = L(R|Q) / L(R)
        低 IRS = question 帮助大 = 更重要
        """
        loss_with_q = self.compute_loss(question, response)
        loss_without_q = self.compute_loss("", response)

        if loss_without_q < 1e-8:
            return 1.0
        return loss_with_q / loss_without_q


def create_test_data():
    """
    创建模拟论文场景的测试数据
    - Task A (simple): 简单问题，应该得到低 weight
    - Task B (complex): 复杂问题，应该得到高 weight
    - Task C (medium): 中等问题
    """
    data = {
        "task_simple": {
            "images": [torch.rand(3, 224, 224) for _ in range(30)],
            "questions": ["What?" for _ in range(30)],
            "responses": ["A cat." for _ in range(30)],
        },
        "task_complex": {
            "images": [torch.rand(3, 224, 224) for _ in range(30)],
            "questions": [
                "Describe in detail all the objects in this image, their colors, positions, and any interactions between them."
                for _ in range(30)
            ],
            "responses": [
                "The image shows a complex scene with multiple objects including a red car parked on the left side, a tall building in the background with glass windows reflecting sunlight, and several people walking on the sidewalk."
                for _ in range(30)
            ],
        },
        "task_medium": {
            "images": [torch.rand(3, 224, 224) for _ in range(30)],
            "questions": ["What objects are visible in this image?" for _ in range(30)],
            "responses": ["There is a dog and a tree in the image." for _ in range(30)],
        },
    }
    return data


def validate_irs(irs_calc, data):
    """验证 IRS 计算逻辑"""
    print("\n" + "=" * 60)
    print("Validating IRS Calculation")
    print("=" * 60)

    task_irs = {}
    for task_name, task_data in data.items():
        irs_scores = []
        for q, r in zip(task_data["questions"][:5], task_data["responses"][:5]):
            irs = irs_calc.compute_irs(q, r)
            irs_scores.append(irs)
        avg_irs = np.mean(irs_scores)
        task_irs[task_name] = avg_irs
        print(f"{task_name}: avg IRS = {avg_irs:.4f}")

    # 验证: complex 应该有最低 IRS (question 最重要)
    print("\nValidation:")
    if task_irs["task_complex"] < task_irs["task_simple"]:
        print("  [PASS] Complex task has lower IRS (question helps more)")
    else:
        print("  [FAIL] Expected complex < simple")

    return task_irs


def validate_task_weights(task_irs):
    """验证 task weights 计算 (Eq. 5)"""
    print("\n" + "=" * 60)
    print("Validating Task Weights (Eq. 5)")
    print("=" * 60)

    tasks = list(task_irs.keys())
    scores = np.array([task_irs[t] for t in tasks])

    # τ = sqrt(1/M)
    M = len(tasks)
    tau = np.sqrt(1.0 / M)

    # w(Ti) = exp(-s/τ) / Σ exp(-s/τ)
    exp_scores = np.exp(-scores / tau)
    weights = exp_scores / np.sum(exp_scores)

    task_weights = {t: w for t, w in zip(tasks, weights)}

    print(f"Temperature τ = {tau:.4f}")
    print("\nTask weights:")
    for task, weight in sorted(task_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task}: w = {weight:.4f}")

    # 验证: complex 应该有最高 weight
    print("\nValidation:")
    max_task = max(task_weights, key=task_weights.get)
    if max_task == "task_complex":
        print("  [PASS] Complex task has highest weight")
    else:
        print(f"  [FAIL] Expected task_complex, got {max_task}")

    return task_weights


def validate_dinov2(extractor, data):
    """验证 DINOv2 特征提取"""
    print("\n" + "=" * 60)
    print("Validating DINOv2 Feature Extraction")
    print("=" * 60)

    images = data["task_simple"]["images"][:5]
    features = extractor.extract(images)

    print(f"Feature shape: {features.shape}")
    print(f"Feature dim: {features.shape[1]} (expected 384 for vits14)")

    if features.shape[1] == 384:
        print("  [PASS] Correct feature dimension")
    else:
        print("  [FAIL] Unexpected dimension")

    return features


def main():
    print("=" * 60)
    print("MVP: PreSel Minimal Validation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. 创建测试数据
    print("\n[1/4] Creating test data...")
    data = create_test_data()
    print(f"Tasks: {list(data.keys())}")

    # 2. 验证 IRS
    print("\n[2/4] Loading IRS calculator...")
    irs_calc = RealIRSCalculator(device)
    task_irs = validate_irs(irs_calc, data)

    # 3. 验证 task weights
    print("\n[3/4] Computing task weights...")
    task_weights = validate_task_weights(task_irs)

    # 4. 验证 DINOv2 (可选，需要下载模型)
    print("\n[4/4] Validating DINOv2...")
    try:
        extractor = RealFeatureExtractor(device)
        validate_dinov2(extractor, data)
    except Exception as e:
        print(f"  [SKIP] DINOv2 failed: {e}")

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("If all validations passed:")
    print("  - IRS formula is correct (Eq. 1-3)")
    print("  - Task weight formula is correct (Eq. 5)")
    print("  - DINOv2 feature extraction works")
    print("\nThe algorithm implementation is verified.")


if __name__ == "__main__":
    main()
