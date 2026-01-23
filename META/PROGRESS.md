# Project Progress: Prototype Selection for 1-NN

## 项目理解

### 核心概念
- **1-NN (1-Nearest Neighbor)**: 给定测试样本，找训练集中最近的点，用它的标签作为预测
- **Prototype**: 从训练集中选出的代表性子集，用于替代完整训练集进行 1-NN 分类
- **Decision Boundary**: 分隔不同类别的边界，由 prototype 的位置决定

### 核心洞察
- 远离边界的点不重要，删掉后边界几乎不变
- **边界点才是关键**：它们决定了分类结果
- 需要平衡：边界覆盖 + 分散性

---

## 算法演进历程

### V1: 初始设计 - Variance-Weighted Boundary Selection

**思路**：
1. 按类内方差分配名额
2. K-Means 聚类保证分散性
3. 每个 cluster 选 boundary score 最高的点

**问题**：太慢！
- Boundary score 需要计算每个点到所有异类点的距离
- 复杂度 O(N²)，60000 个点需要计算 36 亿次距离

### V2: 优化 - 使用 sklearn NearestNeighbors

**改进**：用 Ball Tree 加速最近邻查询
- 预期加速 ~3000x
- 实际：仍然很慢，因为 K-Means 本身是瓶颈

### V3: 进一步优化 - MiniBatchKMeans

**改进**：用 MiniBatchKMeans 替代 KMeans
- n_init: 10 → 3
- 使用 mini-batch 训练

**问题**：仍然需要 7+ 分钟

### V4: 最终版本 - Variance-Weighted Centroid Selection ✓

**关键决策**：放弃 boundary score，简化算法

**最终算法**：
1. 计算每个类别的方差
2. 按方差比例分配名额（方差大的类别分配更多）
3. 对每个类别做 MiniBatchKMeans
4. 选择最靠近每个 centroid 的真实数据点

**为什么这样设计**：
- **去掉 boundary score**：计算代价太高，且 K-Means centroid 本身就能代表类别的典型样本
- **保留方差分配**：不同数字写法变化不同，需要不同数量的 prototype
- **MiniBatchKMeans**：比标准 K-Means 快 5-10x

---

## 设计决策记录

| 决策 | 选项 | 选择 | 原因 |
|------|------|------|------|
| 名额分配 | 平均 vs 按方差 | 按方差 | 写法多变的数字需要更多覆盖 |
| 聚类方法 | KMeans vs MiniBatchKMeans | MiniBatchKMeans | 速度快 5-10x |
| 选点策略 | Boundary score vs 最近中心点 | 最近中心点 | Boundary score 计算太慢 |
| n_init | 10 vs 3 | 3 | 速度 vs 质量权衡 |

---

## 实现进度

- [x] 下载 MNIST 数据集
- [x] 实现算法 (prototype_selection.py)
- [x] 实现 1-NN 分类器 (knn_classifier.py)
- [x] 实现实验框架 (experiments.py)
- [x] Quick test (M=500, 100)
- [ ] 完整实验 (M=10000, 5000, 1000)
- [ ] 撰写 LaTeX 报告

---

## 实验结果 (Quick Test)

### 命令
```bash
cd /Users/waynewang/CSE251A-Project1/src
python main.py --quick
```

### 结果

| M | 我们的方法 | 随机选择 | 提升 |
|---|-----------|---------|------|
| 500 | 91.40% ± 0.46% | 84.70% ± 0.38% | **+6.7%** |
| 100 | 86.19% ± 0.94% | 72.39% ± 1.76% | **+13.8%** |

### 观察
- 我们的方法**显著优于随机选择**
- M 越小，优势越明显（M=100 时提升 13.8%）
- 算法运行速度：约 1-2 分钟完成全部测试

---

## 代码结构

```
src/
├── data_loader.py          # MNIST 数据加载
├── prototype_selection.py  # 核心算法
├── knn_classifier.py       # 1-NN 分类器
├── experiments.py          # 实验运行 & 统计
└── main.py                 # 入口
```

---

## 下一步

- [ ] 运行完整实验：`python main.py`
- [ ] 分析结果，撰写报告
- [ ] 讨论：为什么我们的方法更好？还有改进空间吗？

---

## 参考

- 数据集: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- LaTeX 模板: https://www.overleaf.com/latex/templates/icml2025-template/dhxrkcgkvnkt
