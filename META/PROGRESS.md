# Project Progress: Prototype Selection for 1-NN

## 项目理解

### 核心概念
- **1-NN (1-Nearest Neighbor)**: 给定测试样本，找训练集中最近的点，用它的标签作为预测
- **Prototype**: 从训练集中选出的代表性子集，用于替代完整训练集进行 1-NN 分类
- **Decision Boundary**: 分隔不同类别的边界，由 prototype 的位置决定

### 核心洞察
- 远离边界的点不重要，删掉后边界几乎不变
- **边界区域很重要**，但边界点本身不是好的 prototype
- **典型样本（centroid）才是好的 prototype**：它们能很好地定义决策边界
- 需要平衡：类别覆盖 + 分散性 + 典型性

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

### V5: Cluster-based Boundary Selection ✓

**核心想法**：把 boundary 的思想重新带回来，但用高效的方式实现

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

**结果**：与 V4 准确率几乎相同（差异 < 0.02%），说明 boundary score 排序没有带来提升。

### V6: Boundary-First Selection ✗ (失败)

**核心想法**：直接选择边界点作为 prototype

**算法**：
1. 用 k-NN 识别边界点（邻居中有异类的点）
2. 只在边界点上做 K-Means
3. 选择 centroid 附近的边界点

**结果**：**56.77%** 准确率（比随机的 84.70% 还差！）

**失败原因分析**：
- 边界点往往是噪声样本、模糊样本、非典型样本
- 这些点不是好的 prototype
- **关键洞察**：边界点对于 *定义* 决策边界很重要，但好的 prototype 应该是 **典型的、有代表性的样本**

### V7: Condensed Nearest Neighbor (CNN) ✗ (失败)

**核心想法**：迭代添加被误分类的点

**算法**：
1. 初始化：每类随机选 1 个点
2. 迭代：用当前 prototype 做 1-NN，添加被误分类的点
3. 直到达到 M 个 prototype

**结果**：**62.97%** 准确率（比随机还差）

**失败原因分析**：
- 被误分类的点恰恰是最难分类的点
- 这些点作为 prototype 效果很差

### 核心教训

> **边界点 ≠ 好的 prototype**
>
> 边界点对于定义决策边界很重要，但好的 prototype 应该是**典型样本**（centroid），而不是边界样本。
>
> variance_centroid 效果好正是因为 K-Means centroid 代表的是典型样本。

---

## 设计决策记录

| 决策 | 选项 | 选择 | 原因 |
|------|------|------|------|
| 名额分配 | 平均 vs 按方差 | 按方差 | 写法多变的数字需要更多覆盖 |
| 聚类方法 | KMeans vs MiniBatchKMeans | MiniBatchKMeans | 速度快 5-10x |
| 选点策略 | Boundary score vs 最近中心点 | 最近中心点 | Boundary score 计算太慢 |
| n_init | 10 vs 3 | 3 | 速度 vs 质量权衡 |
| Boundary 计算 | 点级别 vs Centroid 级别 | Centroid 级别 | O(M²) vs O(N²)，快 ~10000x |
| Prototype 来源 | 边界点 vs 全部点 | 全部点 | 边界点是噪声/非典型样本，效果差 |

---

## 实现进度

- [x] 下载 MNIST 数据集
- [x] 实现算法 (prototype_selection.py)
  - [x] variance_centroid (V4) ✓
  - [x] cluster_boundary (V5) ✓
  - [x] boundary_first (V6) ✗ 失败
  - [x] cnn (V7) ✗ 失败
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
- [x] Full 1-NN Upper Bound 实验: 96.91% ✓
- [x] 实验新算法 (boundary_first, cnn) - 失败，记录为 negative results
- [x] 详细趋势实验 (M = 10 ~ 10000) ✓
- [ ] 撰写 LaTeX 报告

---

## 实验结果

### Upper Bound: Full 1-NN

使用全部 60000 个训练样本的 1-NN 准确率：**96.91%**

这是 prototype selection 方法的理论上限。

| Per-class Accuracy | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Full 1-NN | 99.3% | 99.5% | 96.1% | 96.0% | 96.1% | 96.4% | 98.5% | 96.5% | 94.5% | 95.8% |

### 完整实验结果 - Detailed Trend Analysis (5 trials each) ✓

| M | Compression | variance_centroid | random | Improvement | vs Full 1-NN |
|---|-------------|-------------------|--------|-------------|--------------|
| 10000 | 6x | **95.36%** ± 0.13% | 94.76% ± 0.09% | +0.60% | -1.55% |
| 7500 | 8x | **94.96%** ± 0.15% | 94.32% ± 0.11% | +0.64% | -1.95% |
| 5000 | 12x | **94.58%** ± 0.21% | 93.63% ± 0.09% | +0.95% | -2.33% |
| 2500 | 24x | **93.62%** ± 0.19% | 91.96% ± 0.25% | +1.66% | -3.29% |
| 1000 | 60x | **92.47%** ± 0.17% | 88.56% ± 0.51% | +3.91% | -4.44% |
| 500 | 120x | **91.40%** ± 0.33% | 84.70% ± 0.28% | +6.70% | -5.51% |
| 250 | 240x | **89.71%** ± 0.34% | 80.27% ± 0.36% | +9.44% | -7.20% |
| 100 | 600x | **86.19%** ± 0.40% | 72.39% ± 0.80% | +13.80% | -10.72% |
| 50 | 1200x | **82.48%** ± 0.62% | 61.57% ± 4.51% | +20.91% | -14.43% |
| 25 | 2400x | **77.11%** ± 0.91% | 49.76% ± 5.80% | +27.35% | -19.80% |
| 10 | 6000x | **67.01%** ± 0.33% | 39.37% ± 9.21% | +27.64% | -29.90% |

### 关键趋势

```
Improvement vs Random:
M=10000:  +0.60%   ▏
M=7500:   +0.64%   ▏
M=5000:   +0.95%   ▏
M=2500:   +1.66%   ▎
M=1000:   +3.91%   ▌
M=500:    +6.70%   █
M=250:    +9.44%   █▍
M=100:   +13.80%   ██
M=50:    +20.91%   ███
M=25:    +27.35%   ████
M=10:    +27.64%   ████
```

### 运行时间对比

| M | variance_centroid | random |
|---|-------------------|--------|
| 10000 | 234.27s | 0.00s |
| 7500 | 679.60s | 0.01s |
| 5000 | 51.37s | 0.00s |
| 2500 | 315.63s | 0.01s |
| 1000 | 15.60s | 0.00s |
| 250 | 29.15s | 0.00s |
| 50 | 7.84s | 0.00s |
| 10 | 2.40s | 0.00s |

### Negative Results: 失败的算法尝试 (M=500, 2 trials)

| Method | Accuracy | Time | vs Random |
|--------|----------|------|-----------|
| variance_centroid | **91.40%** | 8.67s | +6.7% |
| boundary_first | 56.77% | 292.64s | -28.0% |
| cnn | 62.97% | 10.96s | -21.7% |
| random | 84.70% | 0.00s | baseline |

**分析**：
- boundary_first 找到 5896 个边界点（9.8% 的训练集）
- 边界点是噪声/模糊/非典型样本，不适合作为 prototype
- CNN 迭代添加被误分类的点，但这些点本身就是难分类的样本

### 关键观察

1. **M 越小，优势越明显（核心发现）**
   - M=10000: +0.60% (微小优势)
   - M=1000: +3.91%
   - M=100: +13.80%
   - M=50: +20.91%
   - M=25: +27.35%
   - M=10: +27.64% (巨大优势)

   **结论**：在极端压缩场景下，我们的方法优势最为显著。

2. **极端压缩下仍保持可用准确率**
   - M=50 (1200x 压缩): 82.48% 准确率
   - M=25 (2400x 压缩): 77.11% 准确率
   - M=10 (6000x 压缩): 67.01% 准确率
   - 相比之下，random 在 M=10 时只有 39.37%

3. **随机选择的方差随 M 减小急剧增大**
   - M=10000: random std = 0.09%
   - M=50: random std = 4.51%
   - M=10: random std = 9.21%
   - 我们的方法始终保持低方差 (0.13% - 0.91%)

4. **variance_centroid 和 cluster_boundary 效果相同**
   - 差异 < 0.02%，说明 boundary score 排序没有带来提升
   - 推荐使用更简单的 variance_centroid

5. **推荐的压缩率**
   - 追求高准确率: M=1000 (60x), 92.47%
   - 中等压缩: M=250 (240x), 89.71%
   - 极端压缩: M=50 (1200x), 82.48%

---

## 代码结构

```
src/
├── data_loader.py          # MNIST 数据加载
├── prototype_selection.py  # 核心算法 (5 种方法)
├── knn_classifier.py       # 1-NN 分类器
├── experiments.py          # 实验运行 & 统计
├── visualize_results.py    # 结果可视化
├── run_full_knn.py         # Full 1-NN upper bound 实验
├── quick_test_new_algos.py # 新算法快速测试
└── main.py                 # 入口
```

---

## 下一步

- [x] ~~实现新算法：KNN-Overlap Boundary Selection~~ → 改为 Cluster-based Boundary Selection
- [x] 添加进度条和详细日志
- [x] 模块化重构 (ALGORITHMS registry)
- [x] 完整实验完成 (三种算法对比) ✓
- [x] 尝试新算法 (boundary_first, CNN) → 失败，记录为 negative results
- [ ] 撰写 LaTeX 报告

---

## 参考

- 数据集: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- LaTeX 模板: https://www.overleaf.com/latex/templates/icml2025-template/dhxrkcgkvnkt
