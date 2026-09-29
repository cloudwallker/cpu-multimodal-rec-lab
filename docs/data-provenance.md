# 数据来源、身份与使用边界

核查日期：2026-09-29。本项目使用作者预处理的 Amazon Baby 图文特征和交互，不从网上重新爬取商品原图，也不把随机向量当作真实多模态复现数据。

## 来源入口与下载ID

作者 [FREEDOM 数据说明](https://github.com/enoche/FREEDOM/tree/6f896e244123cfe9baaf7e92a1a3bfc06dfdc3c2/data) 和 [MMRec 数据说明](https://github.com/enoche/MMRec/tree/c68ee7c9b0de1ca7c5150258322edb4994be29d9/data) 指向同一 [Google Drive 总目录](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG)。Baby 子目录ID为 `1Fk21441EO1l7wgOOARh2thu4FjgtKWQp`。以下ID和预期字节数来自下载任务记录，完整性仍以本地文件和manifest核验为准。

| 文件 | Google Drive 文件ID | 来源预期字节数 |
|---|---|---:|
| baby.inter | `1i2IB2bdxu_jMSxr2IvZ04MG54ySgI2JN` | 4362239 |
| image_feat.npy | `103Pxo73naIwmKI6CR71d3t7A5k-lpk9c` | 231014528 |
| text_feat.npy | `1EP-Ro9Lq-RQV_Urrh8Xpa-mvr635tytj` | 10828928 |
| i_id_mapping.csv | `1eagqR_X3H6zFjw1lFXjLLX7fB7AoTxtG` | 111702 |
| u_id_mapping.csv | `1IujVzYfihippYT36v16aNLPpKT4ELwyK` | 392164 |

原开发环境中的五个文件均已完整下载，下载 manifest 记录 `complete=true`。以下 SHA256 为本地计算，并非作者发布的独立校验和；原始数据与 manifest 不随本仓库分发。图像为7050×4096、float64，文字为7050×384、float32。重新下载时由本地生成的 manifest 核验文件哈希与字节数。

| 文件 | 实际SHA256（下载与准备manifest一致） |
|---|---|
| baby.inter | `e0abb033ea5cc538bb2becd8c3dc50b619f28f7974ba61f5ed11ce27cf405940` |
| image_feat.npy | `36c3be592b98506189a7d5de71b21577cf626f0293b539d861534673b3e9fd70` |
| text_feat.npy | `6667f2ad655c9ecc97cb3383f58988864ef51ec0b39c158b15986c66769f2dc4` |
| i_id_mapping.csv | `c56ff96cd1f703b6dc8ac4469856a856019ec1577ac243508e3d253f80878bb6` |
| u_id_mapping.csv | `a80850b8b46008e3bceabdc7ab72ab6e3620616394709daed7d12dc499626b60` |

原开发环境中，2000商品目标经训练清理移除7件商品，最终1993商品、8967用户；train 27977交互，valid 2944交互/2803可评价用户，test 3359交互/3081可评价用户。valid/test长尾正样本为518/576。零特征行排除数为0，但这不证明原生图像完整。本地准备 manifest 的 identity 为 `d5da0ba044736d8a6ecaa2f7831db9007cf9201e8710852db8e6482cb63a173b`；dataset.npz SHA256为 `945710ca695551aee95a23f8c33a613b2e73a9ee9ea7232a03f453b959179b04`。这些是历史数据身份记录；准备数据与 manifest 不随本仓库分发。

## 原生缺图不能由非零矩阵反推

固定快照中 `external/MMRec/preprocessing/3feat-encoder.ipynb` 的图像重索引处理先计算可用特征均值 `avg`，缺少原始图像特征的商品填入该均值。预处理包没有提供对应原生缺图商品名单；不能从没有NaN、没有零行或矩阵行数完整推断每件商品都有原图。[固定版本预处理代码](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/preprocessing/3feat-encoder.ipynb)

因此manifest显式设置 `native_image_availability_verified=false`。项目只校验作者提供的特征矩阵是否有限、行映射是否一致并排除零特征行；`clean` 条件准确含义是“作者预处理特征上不新增人工遮蔽”。`uniform30/tail30` 的结论仅针对本项目人工遮蔽实验。作者均值填充可能使部分可见向量已经是补全值，其数量未知，必须列入限制。

## 哪些身份与哈希应读取manifest

`src/mmrec_lab/data.py` 的准备逻辑针对五个完整原始文件计算SHA256与字节数，并在准备目录生成 `manifest.json`、`dataset.npz`、`item_mapping.csv`、`user_mapping.csv`。准备manifest是实际数据身份的权威记录：

| 字段/产物 | 含义 |
|---|---|
| raw_files | 五个原始文件的实际sha256和bytes |
| raw_shape/raw_dtype | 实际NumPy矩阵维度与dtype，拒绝pickle加载 |
| split_labels | 0=train，1=valid，2=test |
| initial_item_ids/pruned_item_ids/pruning_rounds | 抽样与训练清理的全过程 |
| n_users/n_items/splits | 最终用户商品数、每份交互及尾部正样本数 |
| array_hashes/dataset_sha256/identity | 每个准备数组、NPZ文件及规范化manifest身份 |
| user_ids/item_ids 与映射CSV | 新连续ID到作者原始索引的对应；作者映射文件再连接其原标识 |

不同种子、数据或规则的准备目录不能覆盖既有manifest。加载时重新核对NPZ和数组哈希；遮蔽名单、补全输入与输出、共交互图、模型配置和上游快照应由运行manifest进一步关联。不要只保存“约2000商品”的口头描述，也不要将用户原始标识复制进公开报告。

## 代码许可与数据许可分别记录

当前MMRec代码快照包含 GPL-3.0 许可；本项目CPU模型注明来源、原作者及GPL-3.0-only。独立FREEDOM仓库README有“No commercial use. License reserved by authors.”声明，本地保留为研究参照，其声明不被MMRec许可替换。[MMRec许可快照](https://github.com/enoche/MMRec/blob/c68ee7c9b0de1ca7c5150258322edb4994be29d9/LICENSE)；[FREEDOM说明快照](https://github.com/enoche/FREEDOM/blob/6f896e244123cfe9baaf7e92a1a3bfc06dfdc3c2/README.md)

作者提供下载入口不等于已核实特征、交互及原始Amazon素材可任意再分发。当前未在该五文件数据包中核验独立数据许可，记录为“尚未确认”，不能据代码GPL推断数据也GPL。本公开仓库只提供下载来源、处理流程与不可逆统计身份；原始数据、用户映射和大特征文件不随仓库分发。

本地核验MMRec LICENSE SHA256：`230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809`；FREEDOM README SHA256：`089ccec83f595b5f126c1b62000dc3e19a524062737ea29fe88e87f5e89e9676`。这些哈希记录所阅读声明的身份，不是许可解释或数据授权的替代。
