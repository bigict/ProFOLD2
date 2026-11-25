# 周报：ProFOLD2 多构象 / fold-switching 研究进展

**周期**：2026-11-21 ~ 2026-11-25
**目标**：无条件 fold-switching 系综生成（p(struct|seq) 模态覆盖），蛋白为主。

---

## 一、总体结论（TL;DR）

1. **模型能表示替代 fold**，~2 Å 地板是 FAPE 局部帧损失几何，**非 bug、非容量不足**。
2. **条件化机制成立但有边界**：冻结 trunk + 逐残基 switching 场 adaLN-Zero 可转向替代 fold（XCL1 2.28 Å）；**大幅刚体结构域切换需生成式解码器**。
3. **建成 fold-switcher 挖掘管线**（混合双聚类）：**14,083 switchers**，钙调蛋白类假阳正确排除，全保留 407,890 条目（fold0/fold1 训练底座）。Porter 已知 93 对中 **25 对结构缺库 + 3 对无共同残基（28/93 库不可用），65 对可用**；可用对 swSS 判据上限 **70.8%（46/65）**。
4. **AFDB 增广路线确定**：Foldseek 结构搜索 AFDB（已在 192 核/1.5T 机器下载），per-mode 归类 + 主链-only 构造，喂方向2 FM 解码器。

---

## 二、探针阶段（多构象生成可行性）

**表示能力**（fit-alone，无条件单独拟合）：
- fold0 → 2.56 Å，fold1 → 2.00 Å。**trunk+head 能表示替代 fold**。
- 梯度诊断：trunk/head 梯度健康，**无梯度消失 bug**。
- ~2 Å 记忆地板 = FAPE 局部帧几何（远端残基对误差放大 + clamp），非 lr/clamp/sidechain/容量。fold0↔fold1 相距 5 Å，~2 Å 足以区分。

**条件化机制**（probe v1→v6 迭代）：
- v1/v2 加性 mode 嵌入被 LayerNorm 洗掉（实测存活 <6%）→ 注入点必须在 post-norm。
- **v6 冻结 trunk + 场门控 adaLN-Zero**：XCL1 mode1→fold1 **2.28 Å**，分离 3.3 Å，稳定。
- **trunk 共微调会破坏条件化**（mode1 振荡畸变）→ **冻结 trunk 必要**。
- **可移植性梯度**：XCL1（端部 5 Å）完全成功 > Mad2（拓扑 9.8 Å）部分 6.39 Å > RfaH（结构域 16.8 Å + 54 Å 刚体位移）失败。
- **机制边界**：IPA×8 能把局部条件传播成全局切换，**仅限 IPA 可达范围**；大幅刚体结构域重排需生成式解码器（方向2 FM）。

---

## 三、Fold-switcher 挖掘管线

**方法演进**（匹配口径依赖的召回估计，逐步提升）：

| 方法 | Porter 召回（估计）| 问题 |
|---|---|---|
| APSP 精确 suffix-prefix | ~11.8% | 漏 tag 变体（包含关系非 suffix-prefix）|
| MMseqs2 95% 序列簇 + pairwise swSS | ~32.3% | 代表采样漏具体条目；钙调蛋白假阳 |
| **混合双聚类（最终）** | 见下方硬锚点 | 过分裂簇需 curation |

**召回口径说明（重要）**：上表「召回」是「已知对是否被挖掘标记」的**匹配口径依赖估计**——PDB-id 严格匹配（同簇+不同 fold）与序列簇匹配给出不同值（复核约 38%–72%），**非单一可复现值**，不宜引用精确百分比。**口径无关的硬锚点**（`cross_validation.json` 可复现）：
- Porter 已知 **93 对 = 25 缺库（≥1 结构不在本地 archive）+ 3 无共同建模残基 + 65 完全可用**；即 **28/93（30%）库本身不可用**。
- 可用 65 对直接施加 swSS 判据的**上限 = 46/65 = 70.8%**；另 **15 刚体结构域运动 + 4 小波动按设计排除**（钙调蛋白类，非漏检）。
- 可用对中的挖掘漏检来自跨序列簇（>95% 分歧）/ 同折叠聚类未分裂等，属挖掘层面而非判据上限。

**最终混合框架**：MMseqs2 序列簇 × Foldseek 结构簇（3Di+AA，-c 0.8）+ **簇内 swSS 补充**。
- 序列簇跨 ≥2 Foldseek 折叠簇 = **折叠级 switcher**（RfaH/KaiB/XCL1）；
- 单折叠簇内 swSS 分裂 = **同折叠拓扑 switcher**（Mad2 O/C）；
- **swSS 判据**（switching 残基 SS 变化 ≥0.30 + max_disp≥12）：抓 Mad2 拓扑切换、**排除钙调蛋白刚体运动**（两个盲区都补）。

**结果**：14,083 switchers（折叠级 8976 + 拓扑 5107），全保留 407,890 条目，fold0=多数派/fold1=少数派。

**TM-align 模式对照（弃用）**：更敏感（+4 折叠级切换）但**精度更低**（钙调蛋白假阳，fold_level 绕过 swSS），且慢 5–10×。保持默认 3Di+swSS。

**代码**：`tools/foldswitch_probe/`（分支 feat-foldswitch-mining，commit 42dd755 已推送，待开 PR）。

---

## 四、AFDB 增广（方向2 数据准备）

**路线**：PDB 挖掘 fold0/fold1 模板 → Foldseek 结构搜索 AFDB → per-mode 归类（像 fold0→fold0，像 fold1→fold1）→ 主链-only 构造（§4.3）→ 增广训练数据 → FM 条件生成器。

**数据**：AFDB 已下载于独立机器（**192 核 + 1.5T 内存**，适合 Foldseek 全量搜索），待提供访问。

**Caveat**（§4.2）：AFDB 命中是**伪标签**（预测结构），降权 + identity 门控；结构相似≠保证切换；per-mode 增广 ≠ 新 switcher 发现（新 switcher 需同时匹配两折叠）。

---

## 五、下一步

1. **Curation**：UniProt 同蛋白过滤（砍 ortholog）+ 合并过分裂簇。
2. **AFDB 增广**：提取 fold0/fold1 模板 + Foldseek 搜索脚本（待机器）。
3. **方向2 FM 解码器**：多 fold1 生成（大幅结构域切换 + 多模态）。
4. 挖掘管线开 PR；实验结论补入设计文档。

---

## 附：关键否定结论（避免重走弯路）

- ~~fold0 先验障碍 / 表示受限~~ → 撤回（fit-alone 证明可表示）。
- ~~梯度消失 bug~~ → 排除（梯度健康）。
- ~~lr/clamp_distance/sidechain 导致地板~~ → 排除（FAPE 局部帧几何）。
- ~~APSP+MMseqs 混合提升召回~~ → 否（MMseqs 已覆盖 APSP 全部已知）。
- ~~采样偏差导致漏检~~ → 否（真因：parse_ids 丢链 + 跨簇 + 代表封顶）。
- ~~TM-align 更准~~ → 否（更敏感但精度更低）。