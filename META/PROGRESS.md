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

### V4: Variance-Weighted Centroid Selection ✓

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

### V5: Cluster-based Boundary Selection (新) ✓

**核心想法**：把 boundary 的思想重新带回来，但用高效的方式实现

**原始想法 (KNN-Overlap)**：
- 对每个点找 K 个最近邻
- 如果邻居中有异类，说明靠近边界
- 用 "异类邻居比例" 作为 boundary score
- **问题**：仍然需要对每个点做 KNN 查询，太慢

**优化思路 (Centroid-based Boundary)**：
- 不对每个点计算 boundary score
- 而是对每个 **cluster centroid** 计算 boundary score
- 只需比较 ~M 个 centroid，而不是 N 个点
- 复杂度从 O(N²) 降到 O(M²)

**最终算法**：
1. 按方差分配名额（同 V4）
2. 对每个类别做 K-Means，得到 centroids
3. 对每个 centroid，计算 boundary score = 1 / (到最近异类 centroid 的距离)
4. 按 boundary score 排序 clusters（高分优先 = 更靠近边界）
5. 从高分 clusters 中选点

**直觉**：
- 靠近其他类别的 cluster 更重要（边界区域）
- 远离边界的 cluster 相对不重要
- 优先选边界 cluster 中的点

---

## 设计决策记录

| 决策 | 选项 | 选择 | 原因 |
|------|------|------|------|
| 名额分配 | 平均 vs 按方差 | 按方差 | 写法多变的数字需要更多覆盖 |
| 聚类方法 | KMeans vs MiniBatchKMeans | MiniBatchKMeans | 速度快 5-10x |
| 选点策略 | Boundary score vs 最近中心点 | 最近中心点 | Boundary score 计算太慢 |
| n_init | 10 vs 3 | 3 | 速度 vs 质量权衡 |
| Boundary 计算 | 点级别 vs Centroid 级别 | Centroid 级别 | O(M²) vs O(N²)，快 ~10000x |

---

## 实现进度

- [x] 下载 MNIST 数据集
- [x] 实现算法 (prototype_selection.py)
  - [x] variance_centroid (V4)
  - [x] cluster_boundary (V5)
  - [x] random (baseline)
- [x] 实现 1-NN 分类器 (knn_classifier.py)
- [x] 实现实验框架 (experiments.py)
  - [x] 模块化算法注册 (ALGORITHMS registry)
  - [x] tqdm 进度条
  - [x] 详细输出 (per-class accuracy, prototype distribution)
  - [x] JSON 导出支持
- [x] Quick test (M=500, 100) - variance_centroid vs random
- [x] 完整实验 (M=10000, 5000, 1000) - variance_centroid vs random ✓
- [x] Quick test - 三种算法对比 (variance_centroid, cluster_boundary, random)
- [x] 完整实验 - 三种算法对比 ✓
- [ ] 撰写 LaTeX 报告

---

## 实验结果

### 完整实验结果 - 三种算法对比 (5 trials each) ✓

| M | variance_centroid | cluster_boundary | random | 提升 vs random | 压缩率 |
|---|-------------------|------------------|--------|----------------|--------|
| 10000 | **95.36%** ± 0.13% | 95.34% ± 0.13% | 94.76% ± 0.09% | +0.60% | 6x |
| 5000 | **94.58%** ± 0.21% | 94.56% ± 0.23% | 93.63% ± 0.09% | +0.95% | 12x |
| 1000 | **92.47%** ± 0.17% | 92.46% ± 0.18% | 88.56% ± 0.51% | +3.91% | 60x |

### 运行时间对比

| M | variance_centroid | cluster_boundary | random |
|---|-------------------|------------------|--------|
| 10000 | 234.27s | 267.23s | 0.00s |
| 5000 | 51.37s | 60.37s | 0.00s |
| 1000 | 15.60s | 14.11s | 0.00s |

### Quick Test 结果 (2 trials each)

| M | variance_centroid | cluster_boundary | random | 提升 |
|---|-------------------|------------------|--------|------|
| 500 | 91.40% (9.90s) | 91.38% (6.53s) | 84.70% | +6.7% |
| 100 | 86.19% | 86.32% | 72.39% | +13.8% |

### 关键观察

1. **variance_centroid 和 cluster_boundary 准确率几乎相同**
   - 差异在 0.02% 以内，统计上不显著
   - 两种算法都基于方差分配 + K-Means
   - cluster_boundary 的 boundary score 排序没有带来明显提升

2. **我们的方法始终优于随机选择**
   - 所有 M 值下都有显著提升
   - 95% 置信区间不重叠，差异显著
   - M=1000: +3.91%, M=100: +13.8%

3. **M 越小，优势越明显**
   - M=10000: +0.60%
   - M=5000: +0.95%
   - M=1000: +3.91%
   - M=100: +13.8%

4. **高压缩率下仍保持较高准确率**
   - M=1000 (60x 压缩): 92.47% 准确率
   - 完整 1-NN (~97%) 只损失 ~4.5%

5. **速度对比**
   - 小 M (≤1000): cluster_boundary 略快
   - 大 M (≥5000): variance_centroid 更快 (因为 cluster_boundary 需要排序更多 clusters)
   - Random 最快（无计算开销）

6. **随机选择的方差更大**
   - random std 在 M=1000 时是 0.51%，而我们的方法只有 0.17%
   - 我们的方法更稳定一致

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

- [x] ~~实现新算法：KNN-Overlap Boundary Selection~~ → 改为 Cluster-based Boundary Selection
- [x] 添加进度条和详细日志
- [x] 模块化重构 (ALGORITHMS registry)
- [x] 完整实验完成 (三种算法对比) ✓
- [ ] 撰写 LaTeX 报告

---

## 参考

- 数据集: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- LaTeX 模板: https://www.overleaf.com/latex/templates/icml2025-template/dhxrkcgkvnkt
