# CPU 多模态推荐实验室

### 面向普通 CPU 的 FREEDOM 复现与缺图研究

**使用真实 Amazon Baby 预提取图文特征，运行推荐模型、检查验证排名，并复核已有开发阶段结果。** 工程已实现六种缺图处理及一种按可见邻居数收缩的候选规则；其比较效果尚未得到正式实验证实。

[English](README.md) | 中文

[快速开始](#快速开始) · [现有证据](#现有证据) · [方法](#方法) · [文档](#文档)

![五次已完成开发运行的验证曲线，并非测试集结果](results/interim/validation_curves.png)

*固定 Baby 子集上的五次开发运行验证 Recall@20；不同实验条件分开展示，不能将此图解读为配对方法比较。*

![cpu-multimodal-rec-lab](results/interim/cartoon-infographic.png)

## 现有证据

本仓库交付**可复现工程与阶段报告**，不是已完成的 45 次正式研究。16 个开发配置完成 5 个：4 个 `clean` 骨干候选及 1 个 `uniform30` 固定混合候选。第 6 个运行停止于已保存的第 35 轮检查点；45 次预定主运行尚未开始。当前没有独立测试集成绩，也不能宣称 `support_mix` 有收益。

在 1,993 商品、8,967 用户的准备子集上，选中的 `clean` 骨干于第 85 轮取得**验证 Recall@20 13.3173%、NDCG@20 5.9899%**。这是开发种子 101 的验证选参结果，不是原论文全规模复现或独立测试结果。五次完成运行单次耗时 7.73—18.53 分钟，进程峰值 RSS 660.11—750.90 MiB；测量机器为 i7-12700H、训练使用两个 CPU 线程。详见[阶段报告](docs/findings.md)及[机器可读汇总](results/interim/development_summary.csv)。

已有五个最佳检查点曾独立加载，并重新计算 2,803 名验证用户的 Top-20 排名，与保存记录一致；[核验元数据](results/interim/verification.json)记录这次历史检查。**本仓库不分发原始／准备数据、`best.pt`／`resume.pt` 检查点、逐用户结果或本地环境。** 公开汇总可阅读，但重新运行检查点核验必须自行获取数据并生成本地训练产物。

## 快速开始

需要 Python 3.12 与 Git。以下为 PowerShell 命令；模型训练和评价全程使用 CPU。依赖锁记录实测版本，其中 PyTorch 为 `2.14.0+cpu`。本地环境实际用 `uv` 安装，下面的 pip 安装方式尚未在另一台机器单独验证。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
git clone https://github.com/enoche/MMRec.git external/MMRec
git -C external/MMRec checkout c68ee7c9b0de1ca7c5150258322edb4994be29d9
.\.venv\Scripts\python.exe -m mmrec_lab download
.\.venv\Scripts\python.exe -m mmrec_lab prepare --items 2000 --output data/processed/baby2000
.\.venv\Scripts\python.exe -m mmrec_lab environment
```

下载器获取作者托管的五个 Baby 交互、特征与映射文件，检查大小、格式及本地 SHA256。记录中的目标 2,000 商品经仅依训练交互的清理后为 1,993 件。[数据来源](docs/data-provenance.md)列出下载链接、文件 ID、哈希及数据再分发边界。外部 Google Drive 文件的可用性可能变化。

对照固定上游实现运行正确性测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

先前的主实验前集成测试记录为 76 项通过、30 个子测试通过（JUnit 合计 106 项）；这是历史记录，不代表此处已在新安装环境再测。训练、开发搜索、主实验、报告与基准命令可用 `python -m mmrec_lab --help` 查看，运行时会生成本地结果。`scripts/export_interim.py` **需要未随仓库发布的五个本地运行目录及检查点**，不能仅凭汇总 CSV 重建。

## 方法

FREEDOM 使用固定的图文商品图与去噪用户—商品交互图。CPU 适配以固定提交的 [MMRec FREEDOM 实现](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/src/models/freedom.py)为来源，偏离见[复现对应表](docs/reproduction-map.md)。

候选规则只补全人工遮蔽的图像向量，取至多 20 个训练共交互邻居。令 `n` 为可见邻居数，`μ` 为可见图像向量全局均值，`τ > 0` 待验证集选择：

```text
support_mix = (可见邻居向量之和 + τ × μ) / (n + τ)
```

`n = 0` 时回退 `μ`；隐藏真值和已补全向量都不再作为其他目标的证据。另实现 `zero_fill`、`global_mean`、`paper_neighmean`、`observed_mean`、`fixed_mix`。预定的 `uniform30`、`tail30` 保持相同遮蔽商品数，文字与交互记录不变。既有图补全与均值收缩工作限制新颖性表述，见[相关工作](docs/related-work.md)。作者预处理特征可能已将部分原生缺图填为均值，所以 `clean` 仅指**不新增人工遮蔽**。

## 文档

- [阶段报告](docs/findings.md)：五次验证结果、资源观测、核验范围与未解决的问题。
- [数据来源](docs/data-provenance.md)：作者托管文件、本地哈希、子集身份与再分发边界。
- [复现对应表](docs/reproduction-map.md)、[论文笔记](docs/paper-reading.md)与[相关工作](docs/related-work.md)。
- [有效邻居诊断](docs/support-diagnostics.md)：训练图统计，不是推荐效果。
- [实现说明](docs/implementation.md)：评价、检查点身份与防泄漏规则。

## 署名与许可

模型适配自 Xin Zhou 在 MMRec 的实现，保留 GPL-3.0 与原作者署名，见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。独立 FREEDOM 仓库另有许可声明。代码许可不授予作者预处理 Amazon 数据的再分发权；本仓库不含这些数据或原始图文素材。
