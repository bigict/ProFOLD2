# 工作周报 · TCR-pMHC 特异性评估基准搭建

**日期**：2026-11-24
**项目**：ProFOLD2 / DSW+Fitness
**说明**：本文所有数字均于 2026-11-24 在本机重新核验（CSV 重新解析、基线脚本重跑、磁盘文件清点、PRP 与源矩阵逐格比对），非引用历史记录。

---

## 目标

为 DSW+Fitness 建立与已发表工作**可直接对比**的 TCR 特异性评估集，统一为 ProTCR 输入格式。

## 1. 数据收集（已完成）

从 Woods et al., *Sci Adv* 2026（DOI `10.1126/sciadv.aef5674`）的公开仓库 `AltinLab/tcrtrifold-experiments` 完整抓取 **575 个文件 / 907 MB**，覆盖 cresta、pdb、iedb_I、iedb_II、iedb_meta、test 六组。

仓库中 `.parquet`/`.tsv` 均为 **Git LFS 指针**（约 130 B）。本机无 git-lfs，改为直接调用 **匿名 LFS batch API** 获取签名下载地址，逐个 **sha256 校验**，**0 下载失败、0 残留指针**。

> **关键认识**：论文的最终评估集是 `*/triad/*.conf_af3.parquet`，不是 `raw/*.csv`（后者是过滤前的中间产物）。早期出现的 206 vs 205、130 vs 284 等计数矛盾均源于读错了表。

## 2. 数据集构成（Woods 2026，4 张表）

已导出 4 张 ProTCR 格式 CSV，共 **202,047 行 / 141 MB**。

| 表 | 行数 | 正样本 | 负样本 | p:MHC 抗原 | 类别 | 角色 |
|---|---:|---:|---:|---:|---|---|
| `triads_pdb_validation` | 197 | 16 | 181 | **14** | I | **留出集** |
| `triads_cresta` | 1,433 | 205 | 1,228 | **8** | II | **留出集** |
| `triads_iedb_I` | 165,121 | 8,096 | 157,025 | 1,031 | I | 训练/dev 池 |
| `triads_iedb_II` | 35,296 | 1,668 | 33,628 | 281 | II | 训练/dev 池 |

两个留出集性质不同：

- **`pdb_validation`**：AF3 训练 cutoff（2023-01-12）**之后**发布的 PDB 三元结构，6 个 MHC-I 等位，`MHC_str` 长 274–279 aa。考的是"新结构"。**14 个抗原中 12 个在 cutoff 前的 PDB 里仍有同源匹配**（`pmhc_in_pdb=True`，匹配数 1–32；仅 `LLLGIGILVL`、`NLSALGIFST` 为零匹配）—— 53 个匹配结构的 RCSB 发布日期已逐个核过，全部落在 2001-07-25 至 2022-11-23，无一在 cutoff 之后。论文正文此处写的是 6，与数据表不一致，**本文采用可复算的 12**。该差异不影响泄漏结论：论文同句说明加上 TCR 后没有任何验证 triad 与 pre-AF3 triad 有实质同源（判据为 MHC 与两条 TCR 链均 ≥95%、肽 ≥77%）。
- **`cresta`**：来自 CRESTA 功能实验，**8 个抗原全部无解析结构**，5 种 DQ/DR 异二聚体，`MHC_str` 长 388–391 aa。考的是"全新抗原"，更严格。

**全部 202,047 行 `MHC_str` 非空，0 行会被 ProTCR 的输入过滤丢弃；每个抗原组都同时含正负样本**，per-antigen AUC 处处有定义。

### 2b. 新增第五张表：PRP 激活集（Garcia 组，*Nat Biotechnol* 2026）

Wang N, Yeh H, … **Garcia KC**，*Deep peptide recognition profiling decodes TCR specificity…*，DOI `10.1038/s41587-026-03128-x`，PMID `42129507`。

**为什么收它**：该文**直接把 AF3 ipTM 与 tFold-TCR ipTM 当基线**做 T 细胞激活预测，并被其 pLM 显著超过（AUROC *P*=0.029，AUPRC *P*=0.005，Wilcoxon signed-rank）。与我们的对标任务完全一致。

已导出 `triads_prp_activation.csv`：**336 行 × 18 列**，验收 COMPLETE，**逐格与源矩阵比对 0 处不符**。

| 项 | 值 |
|---|---|
| 构成 | **16 肽 × 21 TCR**，全部 **HLA-B\*27:05**（单一 I 类等位） |
| 标签 | **65 正 / 271 负（19.3%）**，CD69 上调 >3%，TCR 转导 SKW-3，3 次独立实验 |
| `MHC_str` | **真实全长 285 aa**，取自我们自己已验证的 `iedb_I` triad 表，与基准其它表逐字节一致 |
| `TCRA` | **空字符串**（作者未发布 α 序列） |
| `TCRB` | **仅 CDR3β**（14–16 aa），非全长 β 链；`tcr_chain_scope='cdr3b_only'` 逐行标注 |
| 分组 | **按 `tcr_id`**，19/21 可算 AUC（4.1、4.4 全 0）；按肽则 11/16 |

**数据开放度最好**：Dryad `10.5061/dryad.h70rxwdzk` 为 **CC0-1.0**（24 文件 / 184 MB，肽富集计数）；小表在公开 GitHub `akds/PRP-TCR-Specificity`。注意该仓库**无 LICENSE 文件**，代码默认保留所有权利，只取数据不搬代码。

⚠️ **标签语义与前四张不同**：这是**激活**不是**结合**，`label_type='activation_cd69'` 标注，**不可与 IEDB/CRESTA 混池**。论文自身也印证了这点 —— 单 TCR 模型在结合任务上 AUC>0.95，到激活任务只剩 average AUROC **0.64** / AUPRC **0.53**。

### 2c. 导出中发现的一个量化限制：CDR3β-only 的误差下限

**21 个 TCR 只有 16 条不同的 CDR3β**，由此带来两个后果：

1. **主键会撞**：按 `(epitope, 链)` 哈希只得到 256 个唯一 id（336 行）。已把 `tcr_id` 纳入 key，现 **336 个唯一**。
2. **不可约误差**：**15 个 `(CDR3β, peptide)` 输入同时带 0 和 1 标签** → 任何确定性的 CDR3β-only 模型**准确率上限 0.955**（336 中至少错 15）。最极端的 **135.7 与 19.2C1 CDR3β 完全相同，却在 8/16 个肽上标签相反**。

**由此修正一处对论文结论的引用**：该文"α 链无贡献"（α+β vs β-only median PR AUC **0.85 vs 0.86**，*p*=0.39，n=12）是在**结合**任务上测的，**不是激活任务**。上述 8/16 的标签分歧说明，在激活读出上 α 或其它链特征可能仍带信息，**该结论不能直接搬用**。这条警告已写入脚本运行时输出。

## 3. AF3 基线复现（核心结果）

triad 表自带 AF3 置信度特征，因此**无需结构预测、纯 CPU 表格运算**即可复现论文基线。

| 留出集 | 打分特征 | 复现 median AUC | 论文 | 差 |
|---|---|---:|---:|---:|
| pdb_validation（I） | `mean_p_tcr_interface_pae` | **0.920** | 0.92 | −0.000 |
| cresta（II） | `min_p_tcr_pae` | **0.809** | 0.81 | −0.001 |

**指标口径**：每个抗原单独算 ROC，再取抗原间中位数；PAE 类特征取负号。抗原单位为 `(peptide, MHC-I, MHC-II)` 三元组而非单纯 peptide —— 这解释了 13 vs 14、7 vs 8 的计数差异。

**意义**：行数、正负比、抗原数、AUC 四项全部对齐，说明**数据、分组口径、评价函数三者均已验证**。后续 DSW+Fitness 在**同样的 197 / 1,433 行、同样的 14 / 8 个抗原组**上打分，即得到可直接对比的数字。

### 3b. IEDB 全池 AUC（参考锚点，**非论文口径**）

同一脚本加 `--iedb`，对训练/dev 池按 `epitope_id` 分组打分，全量 200,417 行，**3.5 秒**跑完（纯 CPU）。

| 池 | 行数 | 抗原组 | 最佳特征 | median AUC | 范围 |
|---|---:|---:|---|---:|---|
| IEDB I | 165,121 | 1,031 | `mean_p_tcr_interface_pae` | **0.645** | 0.015–1.000 |
| IEDB II | 35,296 | 281 | `mean_tcr_p_pae` | **0.760** | 0.070–1.000 |

13 个候选特征的 median 全部落在 **0.600–0.645（I 类）** 与 **0.640–0.760（II 类）**，即**结论对特征选择不敏感**。

**这组数字与论文的核心警告一致**：论文指出 IEDB 的信号由实验类型主导 —— binding 0.93/0.94、SPR 0.85，而 multimer/tetramer 仅 **0.61/0.52**，且 multimer 占 I 类数据 **80% 以上**。我们不分层的全池 I 类 median **0.645** 正落在 multimer 那一档附近，符合"混合池被低信号实验类型拉低"的预期。

**必须标注的四点口径差异**（脚本运行时会自动打印同样警告）：

1. 分组用 `epitope_id`（1,031 / 281），论文用 **antigen-study**（1,024 / 248）；表内 `references` 在置换负样本上是占位符，无法重建该分组。
2. 表内**无 assay 字段**（78 / 79 列均无），无法复现论文的分层结果。
3. 行集为**同源排除前**，正样本多 394 / 43，负样本完全一致。
4. IEDB 是论文的**训练/dev 池**，其模型 in-sample；我们的模型若未训过 IEDB 则是真留出，两边不是同一个实验。

**因此：IEDB 的 0.645 / 0.760 只能作为我们自己的参考锚点，不能写成"复现论文"。** 真正复现的只有 §3 的两个留出集。

## 4. 交付与验证

- 数据与脚本已传至计算服务器 `ProTCR_data/woods2026/`，六文件 md5 一致。
- **以服务器重跑为准**：远端 `python3`（pyarrow 23.0.0）复现出 **0.920 / 0.809**，与本机（26.0.1）完全一致；仅从导出的 CSV 重算得 0.9199。结果不依赖机器、reader 版本或 parquet/CSV 路径。
- 四个脚本（`collect.py` / `normalize.py` / `baseline_auc.py` / `prp_activation.py`）已提交至分支 **`tcr-bench` @ `7bb5025`** 作为代码备份，不开 PR、不合入 main。三次提交均为快进：`9cfc0f2` → `41bf131`（IEDB 打分 + 本周报）→ `7bb5025`（PRP 导出）。每次推送后均核对 `origin/main` 仍为 `94e03c6` 未被触碰，且该分支无 PR。
- 数据、`csv/` 与 venv 由 `.gitignore` 排除，可完全重建；五张 CSV 全部不入库。

## 5. 已知问题 / 下周

1. **II 类 MHC 编码未验证**：ProTCR 只接受单条 `MHC_str`，CRESTA 全为 II 类，目前用 α+β 拼接（388–391 aa）。能否通过 `aligned_ratio ≥ 0.85` 过滤**尚未实测**，是跑通前的首要风险。
2. **基线特征身份为数值吻合，非确认**：论文可达文本未点名具体特征，扫描 14 个候选后取最接近者。
3. **IEDB 正样本数与论文有差**：8,096 / 1,668 vs 论文 7,702 / 1,625（差 394 / 43），而**负样本数完全一致**（157,025 / 33,628）。差异只在正样本侧，疑为论文的同源排除步骤，未证实。
4. **第二篇文献的 TCRvdb 已定位但受限**：Messemaker, *Immunity* 2026（DOI `10.1016/j.immuni.2026.09.007`）。GitHub 仓库 `schumacherlab/TCRvdb` 仅含 README，数据在 `tcrvdb.nkiai.nl`，需实名申请（Email / Name / Organization / Purpose + captcha）与人工审核，下载链接 7 天单次有效，且**明确禁止再分发**，不能沿用现有的共享服务器流程。其价值在于标签经过功能验证 —— 公开库中仅约 56% 的 pair 能复现标注反应性。发布版覆盖 5 个 epitope（YLQ / GLC / TTD / GIL / NLV）。
5. **PRP 缺 TCR 全长**：GitHub 与 Dryad 都只给 CDR3β，`TCRA` 只能留空。若要跑完整 ProTCR 输入，需从 Supplementary Table 1 或作者处补全长 α/β。在补齐前，该表的结果受 **0.955 准确率上限**约束（见 §2c）。
6. **第三篇已排除**：NeoDisc（Huber et al., *Nat Biotechnol* 2024，DOI `10.1038/s41587-024-02420-y`）。四张补充表逐单元格扫描 `CDR`/`TRAV`/`TRBV`/`TCR` **命中 0**；它做的是**肽层面新抗原优先级排序**，不是 TCR 特异性。S4 仅 63 条肽 / 5 个病人，且文章为 **CC BY-NC-ND 4.0**（禁止分发改编材料）。其唯一可用切片是 12 条 tumor-rejection 验证肽，但属另一任务，且其筛选引用 IEDB，与我们已有池重叠，不构成独立留出集。

## 附：数据完整性的一处修正

此前"575 个文件全部拿到"的表述不够准确。复查发现其中 **3 个是上游仓库的符号链接**（git mode `120000`），指向作者私有路径 `/scratch/lwoods/...`，抓取后只是约 80 字节的路径文本，无法解析：

```
data/pdb/pmhc/staged/pdb_pmhc.cleaned.parquet
data/pdb/triad/staged/pdb_triad.cleaned.parquet
data/pdb/triad/staged/pdb_triad.conf_boltz.parquet
```

三者都在 `staged/` 中间目录下，**均非我们读取的表**，不影响任何结果。准确表述是 **572 个真实数据文件 + 3 个不可解析的上游符号链接**。

另，`manifest.json` 每次运行会被覆盖，当前只记录最后一批 373 个，统计总数应以磁盘实际文件为准。

---

## 复现命令

```bash
# 1. 收集（stdlib only，无需 git-lfs）
python3 tcr_bench/collect.py

# 2. 导出 ProTCR CSV（需 pyarrow）
python3 -m venv tcr_bench/.venv
tcr_bench/.venv/bin/pip install pyarrow
tcr_bench/.venv/bin/python tcr_bench/normalize.py --mhc2 concat --iedb

# 3. 复现 AF3 基线 AUC（两个留出集）
tcr_bench/.venv/bin/python tcr_bench/baseline_auc.py

# 3b. 追加 IEDB 全池参考锚点（非论文口径，脚本会打印警告）
tcr_bench/.venv/bin/python tcr_bench/baseline_auc.py --iedb

# 4. 导出 PRP 激活集（需先完成步骤 1，MHC 序列取自 iedb_I triad 表）
tcr_bench/.venv/bin/python tcr_bench/prp_activation.py
```

五张表的口径速查：

| 表 | 行数 | 标签 | 分组键 |
|---|---:|---|---|
| `triads_pdb_validation` | 197 | 结合 | `epitope_id`（14） |
| `triads_cresta` | 1,433 | 结合 | `epitope_id`（8） |
| `triads_iedb_I` | 165,121 | 结合 | `epitope_id`（1,031） |
| `triads_iedb_II` | 35,296 | 结合 | `epitope_id`（281） |
| `triads_prp_activation` | 336 | **激活 CD69** | **`tcr_id`（19 可用）** |
