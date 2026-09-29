# 缺失图像补全查新与基线依据

核查日期：2026-09-29。对象：FREEDOM CPU 实验中的共交互 Top20 可见邻居均值收缩。本文是有范围限制的查新记录，不是原创性证明；尚无本项目实测结论。

## 1. 结论与允许的表述

**共交互图补全、单步邻居均值、训练前补全再使用 FREEDOM，以及邻域估计向全局均值收缩，都已有相关先例。** 本次核查的五个主要来源中，没有确认与下面整个实现约定完全相同的已发表方法；这不意味着它是新算法。

项目应定位为“FREEDOM 的 CPU 缩小规模复现及按可见邻居支持量收缩的轻量补全实验”。可以称候选微改或已有收缩思想的特定应用，不能称首个缺失模态方法、原创图补全算法或已具备投稿创新。uniform30 与 tail30 是本项目的场景对照，不是独立算法贡献。

## 2. 五个主要一手来源

| 来源与日期 | 实际核查位置与覆盖范围 | 对本项目的含义 |
|---|---|---|
| Malitesta 等，*Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation*；arXiv v1：2026-02-19；摘要页注明已被 IEEE TKDE 接收。[版本与日期](https://arxiv.org/abs/2602.17354v1)；[全文](https://arxiv.org/html/2602.17354v1) | IV-B：共交互 TopN 图；Eq.10 缺失置零；Eq.11 NeighMean 按图度数归一化；Eq.12–14 多跳及扩散。V-B 包含 FREEDOM、Zeros、GlobalMean。 | 最接近的已有工作；论文 Eq.11 未加入本项目的全局均值伪计数项。 |
| 同作者，*Do We Really Need to Drop Items with Missing Modalities in Multimodal Recommendation?*；arXiv v1：2024-08-21；CIKM 2024 短文。[版本与日期](https://arxiv.org/abs/2408.11767v1)；[全文](https://arxiv.org/html/2408.11767v1) | §2.2 的 NeighMean、MultiHop、PersPageRank；§3 与 Table 4 已在 FREEDOM 上比较补全方法。 | 前述路线至少在 2024 年已明确出现，不能把 2026 文献当作路线起点，也不能把“接入 FREEDOM”当创新。 |
| Rossi 等，*On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs With Missing Node Features*；LoG：2022-12-09 至 12；预印本：2021-11-23。[会议原文与 PDF 入口](https://proceedings.mlr.press/v198/rossi22a.html)；[预印本日期](https://arxiv.org/abs/2111.12128) | 从 Dirichlet 能量推导扩散式 Feature Propagation，针对缺失节点特征。 | 缺失特征通过图传播恢复有更早的一般方法基础；它不是本项目的单步全局均值收缩。 |
| Wilcox、Giagos、Djahel，*A Neighborhood-Similarity-Based Imputation Algorithm for Healthcare Data Sets: A Comparative Study*；2023-11-28，Electronics 12(23):4809。[论文 DOI](https://doi.org/10.3390/electronics12234809)；[作者机构存档全文](https://repository.essex.ac.uk/37298/1/electronics-12-04809.pdf) | §3.3 Eq.3 将相似邻域产生的候选补全与可观测全局均值混合；权重由局部/全局方差估计驱动，并给出经验贝叶斯解释。 | 直接覆盖“邻域补全向全局均值收缩”的思想；其任务是表格医疗数据，候选生成与权重均不同，不能据此认定整个项目规则已完全相同。 |
| Li 等，*Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion*（GRE-MC）；arXiv v1：2026-05-01。[版本与日期](https://arxiv.org/abs/2605.00670v1)；[全文](https://arxiv.org/html/2605.00670v1) | §3.2：模态感知子图检索、Graph Transformer 联合编码与稀疏路由码本。 | 截至核查日，同领域已有超出邻居聚合的补全框架；本项目只研究受 CPU 约束的轻量方法，不声称达到近期方法水平。 |

## 3. Eq.11 与项目规则的精确区别

记训练共交互图中 Top20 的正权重、非自身邻居为 `N_i`，图度数 `d_i=|N_i|`，可见图像集合为 `O`，可见邻居数 `n_i=|N_i∩O|`，可见全局均值为 `mu`。

依据 2026 论文 Eq.10–11，缺失向量置零后，对 `d_i>0` 有：

```text
paper_neighmean_i = sum(v_j, j in N_i∩O) / d_i
```

这一步是从论文矩阵式进行的代数展开。项目拟用：

```text
observed_mean_i = sum(v_j, j in N_i∩O) / n_i      # n_i>0
support_mix_i = (sum(v_j, j in N_i∩O) + tau*mu) / (n_i+tau)
```

故 `paper_neighmean=(n_i/d_i)*observed_mean`。例如四个图邻居只有一个可见，论文式为该可见向量的四分之一；可见均值为该向量本身。**不能将 observed_mean 直接标作 Eq.11 的严格复现。** `d_i=0` 的除零处理需明确记录；在项目基线中采用零向量可以保持“没有邻居证据”的语义，但这是本地约定，不能声称论文公式规定了该分支。

可见均值与候选收缩在 `n_i=0` 时均回退 `mu`；因此二者的比较控制了零支持回退，主要检验正支持情况下的收缩。候选仅使用原始可见图像，一次完成，不递归使用已补全向量。

## 4. 收缩规则的数学地位

下面是本报告的独立推导，而不是引述某篇论文的新定理。对固定 `mu` 和 `tau>0`，最小化

```text
L(z) = sum(||z-v_j||², j in N_i∩O) + tau*||z-mu||²
```

其唯一解就是项目规则。它是带全局均值目标的二次正则化局部均值，也可以将 `tau` 理解为全局均值的伪支持量；不应把这个代数形式本身作为复杂算法创新。

在额外假设下，若邻居向量独立服从均值 `theta_i`、协方差 `sigma² I` 的高斯观测，且 `theta_i` 的先验为均值 `mu`、协方差 `s² I` 的高斯分布，则后验均值同形，`tau=sigma²/s²`。真实共交互邻居不保证独立、同分布或同方差；本项目的 `tau` 由验证集选择，`n_i` 只是支持量代理，不能声称权重是已校准的置信度或已证明最优的贝叶斯估计。可见全局均值还是本场景的样本估计，并非已知真先验。

## 5. 检索覆盖与未证明事项

实际检索词包括 `multimodal recommendation missing modality shrinkage`、`support-aware imputation`、`degree-aware missing features`、`imputation neighbor shrinkage mean global`、`Bayesian neighborhood imputation mean` 与 `feature propagation missing Rossi 2022 arxiv`；并沿最近论文核对前作。搜索摘要仅作为线索，核心判断使用上表全文或作者/会议原文。

已覆盖：同领域最接近的训练前图补全公式、FREEDOM 先例、一般图缺失特征传播、邻域补全向全局均值收缩，以及一个 2026 年更复杂的图补全框架。

未证明：是否已有未被索引、换名发表或代码中实现的 `n/(n+tau)` 可见邻居图像补全；是否已有相同 tail30 协议；所有 support-aware/degree-aware 领域方法是否包含等价规则。检索到度感知、长尾邻域信息补充等相邻方向，不足以仅凭标题判定公式相同或不同。本次未全面审计这些方法的代码和补充材料，也没有完成全领域系统综述。不能以精确关键词无命中、公式有小差别或未见同名方法推断新颖性。

## 6. 本期基线与评价理由

| 基线/对照 | 理由与实现约定 |
|---|---|
| clean | 作者预处理特征上无新增人工遮蔽的参考条件；不保证全部商品存在原生图像，也不保证指标最高。 |
| zero_fill | 测量缺失信息的朴素处理结果；缺图商品输入零向量。 |
| global_mean | 测量全局统计本身的作用；只由当前场景仍可见原始图像计算。 |
| paper_neighmean | 最近论文的必要对照：分母 `d_i`，缺失邻居计零；孤立项明确采用本地零回退约定。 |
| observed_mean | 排除仅仅改变均值分母带来的收益；分母 `n_i`，零支持回退 `mu`。 |
| fixed_mix | 用验证集选择固定 `alpha`，计算 `alpha*observed_mean+(1-alpha)*mu`；检验是否必须按支持量变化。实际搜索 `{0.25,0.5,0.75}`；两端由 global_mean/observed_mean 独立覆盖，零支持回退 `mu`。 |
| support_mix | 候选规则；预先固定 `tau∈{1,4,16}`，仅用验证集选择，记录全部候选与选择结果。 |

上述基线由 Eq.11 的区别和本项目研究问题决定。MultiHop 等属于后续可选更强图基线；本期若未实现，报告必须承认未覆盖，不能声称胜过全部图补全或同领域 SOTA。

uniform30 与 tail30 应保持缺图商品总数相同，训练交互图及数据划分不变；热度与 Top20 图只用训练交互，遮蔽向量不能参与补全。分别报告整体、预定义长尾、缺图目标的推荐指标及人数/正样本量，并按 `n_i` 分组诊断。长尾缺失不等于真实零交互冷启动，也不能自动命名为已验证的 MNAR 机制。

只有在同掩码、同训练预算与配对种子下优于可见均值及验证集选出的固定混合，并报告不确定性、CPU 成本和失败分组，才能支持“该支持量规则在本实验范围有用”。即使获得稳定提升，结论仍是该组合的实证结果；若找到完全相同规则，应改为已有方法复现，删除创新声明。
