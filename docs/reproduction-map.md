# 原论文、官方实现与本项目对应

核查日期：2026-09-29。本项目是 CPU 缩小规模复现与缺图扩展，不是原论文全套指标复现。阅读 [FREEDOM v3](https://arxiv.org/html/2211.06924v3)；实现参照固定版本的 [MMRec FREEDOM](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/models/freedom.py)。

## 来源快照

| 仓库 | 本地核验提交 | 模型文件 SHA256 |
|---|---|---|
| enoche/MMRec | `c68ee7c9b0de1ca7c5150258322edb4994be29d9` | `b5bbbf0a5e04c347bcb508bfdfb5351aec9417caba1fabf151cd55825e9dd732` |
| enoche/FREEDOM（研究参照） | `6f896e244123cfe9baaf7e92a1a3bfc06dfdc3c2` | `899ae3cf069e7dc8dddb998af890dfb8383d716d46a84424de5af52430b09838` |

以上由本地 `git rev-parse HEAD`、`Get-FileHash` 核验。模型文件整体哈希不同；当前计算代码一致，文件顶部说明不同。MMRec 带 GPL-3.0 许可；FREEDOM 单独仓库 README 写“No commercial use. License reserved by authors.”，两者不能混为一种许可。本适配的代码来源和署名见 `src/mmrec_lab/model.py`，具体实现记录见 [implementation.md](implementation.md)。

## 四列映射

| 原论文 | 官方实现 | 本项目 | 偏离/边界 |
|---|---|---|---|
| §3.1 Eq.1–3 原始特征余弦、二值 kNN、度归一化、模态混图 | `get_knn_adj_mat`；k=10；图像权重0.1 | `FreedomModel._knn_graph` 和 `mm_adj` | 内容图保留自邻居、有向 Top-k；零范数行排除查询与候选，官方直接除范数可能产生 NaN。 |
| 初始化后固定商品图 | 保存/加载语义图缓存 | 每次由本次补全输入重新构图 | 不沿用粗粒度缓存，防止不同掩码和方法复用旧图；图固定而特征表可训练。 |
| §3.2 Eq.4 用户—商品图与度敏感剪枝 | `get_norm_adj_mat/pre_epoch_processing` | `_norm_ui_graph/sample_epoch_graph` | CPU 独立随机流；零保留边时空图。原论文保留边数为ceil，官方与本实现int即floor，可能差一条边。 |
| §3.3 Eq.5–8 商品图传播、交互层均值、表示相加 | `forward` | `forward` | 保留语义1层、交互2层等默认结构；研究配置若改变以运行manifest为准。 |
| Eq.9–10 线性投影与主/辅助 BPR | `calculate_loss/bpr_loss` | `loss/_bpr` | 只投影批次正负商品，逐行线性等价于整表投影；采用稳定logsigmoid。 |
| §3.4 Eq.11 最终内积评分 | `full_sort_predict` | `scores` | 推理用完整训练图，评分不额外直接相加投影模态向量。 |
| 完整Baby/Sports/Clothing与论文实验设置 | 作者预处理包、配置、训练器 | `prepare_dataset` 固定训练分层子集 | 首期约2000商品目标，经训练清理可减少；只做一个数据集，不能对照论文数字宣称涨跌。 |
| 未以支持量收缩作为 FREEDOM 核心 | 2024/2026 图补全论文另行提出训练前补全 | `build_neighbors/impute_features` | 共交互Top20图是扩展，区别于内容kNN=10；公式与基线见 [related-work.md](related-work.md)。 |
| 论文标准推荐评价 | 官方topk评估模块 | `metrics.evaluate/evaluate_rankings` | 项目增加长尾、缺图及支持量诊断；组内指标保留全目录名次，不重排组内商品。 |

## 数据与训练协议的偏离

作者标签 `x_label=0/1/2` 对应 train/valid/test；沿用标签，不重新分割或把过滤掉的测试记录转入训练。只根据训练热度分四层抽样，固定种子20260929；反复删除训练不足两次的用户，再删除无训练商品。长尾为最终训练热度排序后半商品，热度相同时原始商品ID稳定排序。实际集合、删减轮数与哈希进入准备manifest。

`uniform30` 与 `tail30` 均遮蔽 `floor(0.3*N)` 商品，后者只从长尾集合抽取；同一次种子下各方法共享名单。缺图行在进入补全接口前已置零；可见全局均值、Top20邻居、图文内容图均不读取隐藏真值。`clean` 是预处理包上的“无新增人工遮蔽”，没有证明全部原生图像存在。

CPU 预算、epoch、batch、早停、选择指标和学习率的最终值必须以运行配置为准。本文件记录源码可核验行为，不替执行器预先声称最终数值。官方默认GPU、1000 epoch与参数网格不会自动代表本项目实际执行预算。

## 验证证据如何阅读

历史测试曾导入固定上游快照，比较图、前向、评分、剪枝、总损失与参数梯度；实现约定见 [implementation.md](implementation.md)。目前真实数据的验证结果与完整研究的边界见 [阶段报告](findings.md)。
