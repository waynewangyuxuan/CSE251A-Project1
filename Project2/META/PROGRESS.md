# Progress Log — Project 2: Coordinate Descent

> 所有进展记录在此，后续写论文直接从这里整理。

---

## 2026-02-09: 项目初始化 & 全部实验完成

### 数据
- Wine dataset, class 1 & 2 only: 130 points, 13 features
- 标准化后使用，augment 一列 1 作为 bias → 14 维

### Baseline
- sklearn LogisticRegression(C=1e10, 无正则化), accuracy = 1.0000
- **L* = 0.00029259**
- 注意：数据线性可分，无正则化的 loss 理论下确界为 0，L* 是 sklearn solver 停下来的位置
- 我们的 CD 方法会继续优化到 loss ≈ 0，这是正确行为

### 算法设计

**选定方案: Momentum + Newton**

**(a) 坐标选择 — Momentum-weighted Gauss-Southwell:**
- 维护每个坐标梯度绝对值的 EMA (exponential moving average)
- Score = |∂L/∂w_j| + 0.5 × EMA_j
- 选 score 最大的坐标
- 好处：避免在两个坐标间震荡，利用历史信息

**(b) 坐标更新 — Newton step:**
- w_j -= g_j / H_jj （一维 Newton 更新）
- g_j = 偏导数, H_jj = Hessian 对角元素
- clip step 到 [-10, 10] 保证稳定性
- 需要 L(·) 二阶可微

### 实验结果：6 种组合对比

| 方法 | Final Loss (2000 iter) |
|------|----------------------|
| **GS + Newton** | ~0 (1e-12) |
| **Momentum + Newton** | ~0 (1e-12) |
| Random + Newton | ~0 (1e-12) |
| GS + Adam | 0.00203 |
| Momentum + Adam | 0.00200 |
| Random + Adam | 0.00611 |

**关键发现：**
- Newton 更新远优于 Adam（二阶信息 vs 一阶自适应）
- GS 和 Momentum 选择策略收敛速度相近，都 ~300 iter 收敛
- Random 选择 ~1000 iter 才收敛（慢 3-4x）
- Adam 在 2000 iter 内未收敛到 L*

### Sparse Coordinate Descent 结果

策略：active set + swap 机制
- 维护 ≤k 个非零特征坐标（bias 不计入）
- 预算未满时：激活梯度最大的 inactive 坐标
- 预算已满时：如果最佳 inactive 梯度 > 最弱 active 梯度，则 swap

| k | Loss | Nonzero | Active Features |
|---|------|---------|-----------------|
| 1 | 0.3636 | 1 | [0] |
| 2 | 0.1575 | 2 | [0, 12] |
| 3 | 0.1458 | 3 | [0, 3, 12] |
| 4 | 0.1141 | 4 | [0, 3, 11, 12] |
| 5 | 0.0542 | 5 | [0, 2, 3, 11, 12] |
| 7 | 0.0012 | 7 | [0, 1, 2, 3, 10, 11, 12] |
| 9 | 0.0002 | 9 | [0, 1, 2, 3, 6, 9, 10, 11, 12] |
| 11 | ~0 | 10 | [0, 1, 2, 3, 4, 6, 9, 10, 11, 12] |
| 13 | ~0 | 11 | [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12] |

- k=9 时 loss 已接近 L*
- k=11 时达到 full model 精度（实际只用了 10 个特征）
- 贪心策略不保证全局最优 k-sparse 解（即使 L 凸），因为特征子集选择是组合优化问题

### 生成的图表
- `figures/all_methods_comparison.pdf` — 6 种方法对比
- `figures/adaptive_vs_random.pdf` — Momentum+Newton vs Random+Newton
- `figures/sparse_loss_vs_k.pdf` — Loss vs sparsity budget k
