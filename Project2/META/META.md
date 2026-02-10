# Project 2: Coordinate Descent — 工作流

## 项目结构

```
Project2/
├── META/
│   ├── META.md          # 工作流 & 注意事项
│   ├── PROGRESS.md      # 所有进展记录（写论文用）
│   └── TODO.md          # 当前任务清单
├── src/                 # 所有源代码
│   ├── data_loader.py   # Wine数据加载 & 预处理
│   ├── coordinate_descent.py  # 核心算法实现
│   ├── experiments.py   # 实验运行脚本
│   └── visualize.py     # 画图
├── figures/             # 生成的图表
├── results/             # 实验结果数据
├── Report/              # LaTeX报告
└── REQUIREMENT.pdf
```

## 工作流

1. **干活前**：看 TODO.md，确认当前任务
2. **干活中**：代码全放 `src/`
3. **干活后**：把进展、实验结果、设计决策写进 PROGRESS.md
4. **最后**：从 PROGRESS.md 整理写 LaTeX 报告

## ⚠️ 重要注意事项（来自 Requirement）

### 数据
- Wine 数据集：178 点，13 维，3 类
- **只用 class 1 和 class 2**（59 + 71 = 130 个点）
- 数据来源：https://archive.ics.uci.edu/ml/datasets/Wine

### 实验要求
- **Baseline**: sklearn LogisticRegression，**不加正则化**（C 设很大）
  - 记录最终 loss L*
- **对比实验**: 我们的方法 vs random-feature coordinate descent
  - random 版本：坐标随机选，更新方式和我们一样
- **图表**: L(wt) vs t，两条曲线，asymptote 到 L*

### 报告要求（5 个部分）
1. 算法描述：如何选坐标 + 如何更新，是否需要可微性
2. 收敛性分析（不需要证明，简要解释即可）
3. 实验结果 + 图表
4. 批判性评价：改进空间
5. **Sparse coordinate descent**:
   - k-sparse 版本（最多 k 个非零元素）
   - 凸函数下能否找到最优 k-sparse 解？
   - Wine 数据上测试，loss vs k 的表格

### 提交格式
- **LaTeX 报告**（ICML 2025 conference style，和 Project 1 一样）
- **代码 zip 文件**（单独上传到 Gradescope）
