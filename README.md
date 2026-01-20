# PreSel 复现

论文: "Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning" (CVPR 2025)


```bash
pip install -r requirements.txt
python3 tests/test_presel.py
```

## 文件结构

```
presel_project/
├── presel/                      # 核心代码
│   ├── presel.py               # 主流程，PreSel 类和 baseline
│   ├── irs_calculator.py       # IRS 计算 (Eq.1-3)，有简化版和完整版
│   ├── task_importance.py      # task importance 估计 (Eq.4-5)
│   ├── cluster_selector.py     # k-means 聚类 + NC score 选样本 (Eq.6-7)
│   ├── visualization.py        # 画图用的
│   └── data/
│       ├── dataset.py          # 数据集加载 (LLaVA/Vision-Flan/dummy)
│       └── utils.py            # 保存/读取结果
│
├── tests/                       # 测试
│   ├── test_install.py         # 检查依赖装好没
│   └── test_presel.py          # 跑一遍完整流程
│
├── mvp/                         # 最小验证
│   ├── download.py             # 下载 DINOv2 和样本数据
│   └── validate.py             # 验证公式实现对不对 (DINOv2 + GPT2)
│
├── run_presel.py               # 主脚本，跑真实数据用这个
├── setup.sh                    # 环境安装
└── requirements.txt
```

## 用法

```bash
# dummy 数据测试
python3 run_presel.py --data_type dummy --num_tasks 5 --samples_per_task 100

# 真实数据 (LLaVA format)
python3 run_presel.py --data_type llava --annotation_file xxx.json --image_dir ./images
```

## 算法

1. 随机采 5% 数据算 IRS，得到每个 task 的 weight
2. 用 DINOv2 提特征，k-means 聚类
3. 按 Neighbor Centrality 从每个 cluster 选代表性样本
4. 最终选 15% 数据，效果接近用 100% 数据训练

## 注意

- `presel/irs_calculator.py` 里的 `SimplifiedIRSCalculator` 是启发式近似，不是论文的真实实现
- 真实实现需要 LLaVA 模型算 loss，见 `IRSCalculator` 类
- DINOv2 特征提取也是用 dummy 的，真实版需要加载模型

## 验证状态

- [x] 公式实现 (Eq.5, 6, 7)
- [x] DINOv2 特征提取 (mvp/validate.py 验证过)
- [ ] IRS 语义验证 (需要 LLaVA，没做)
- [ ] 在 LLaVA-1.5/Vision-Flan 上复现论文数值

## 引用

```bibtex
@inproceedings{safaei2025presel,
  title={Filter Images First, Generate Instructions Later},
  author={Safaei, Bardia and Siddiqui, Faizan and Xu, Jiacong and Patel, Vishal M. and Lo, Shao-Yuan},
  booktitle={CVPR},
  year={2025}
}
```
