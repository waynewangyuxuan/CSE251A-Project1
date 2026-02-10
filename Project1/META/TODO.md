# TODO List

## 算法改进

### ~~新算法想法：KNN-Overlap Boundary Selection~~ ✓ 已实现 (改为 Cluster-based)

**原始思路**：
- 对每个点找 K 个最近邻（在所有训练数据中）
- 如果邻居中有异类，说明该点靠近边界
- 用 "异类邻居比例" 作为 boundary score
- **问题**：仍然 O(N * K * D)，太慢

**最终实现：Cluster-based Boundary Selection**
- 对每个 **centroid** 计算 boundary score = 1 / (到最近异类 centroid 的距离)
- 只比较 ~M 个 centroid，复杂度 O(M²)
- 优先从高 boundary score 的 cluster 中选点

**状态**: ✅ 已实现 (`select_prototypes_cluster_boundary`)

---

## 工程改进

### 1. 进度条 ✅
- [x] 添加 tqdm 进度条
- [x] 显示：当前 M 值、当前 method、trial 进度

### 2. 详细输出 & 日志 ✅
- [x] 每个实验保存详细结果到 JSON (--output-dir)
- [x] 记录：
  - 每个 trial 的准确率 (all_accuracies)
  - 选择时间、预测时间
  - 每个类别的 prototype 分配数量 (prototype_distribution)
  - 每个类别的分类准确率 (per_class_accuracy)
- [ ] 支持实验结果可视化 (低优先级)

### 3. 模块化实验框架 ✅ (简化版)
- [x] ALGORITHMS registry 支持动态注册算法
- [x] 命令行参数支持 (--methods, --M, --n-trials)
- [ ] 完整插件式架构 (低优先级，当前实现已足够)

---

## 报告 & 文档

- [ ] 撰写 LaTeX 报告
- [ ] 绘制结果图表
- [ ] 整理实验数据

---

## 优先级

1. ~~**高**：等三种算法完整实验跑完~~ ✅ 已完成
2. **高**：撰写 LaTeX 报告
3. **低**：实验结果可视化（如果时间允许）
