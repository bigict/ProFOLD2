# 无条件 Fold-Switching 构象系综采样 — 设计方案

> 状态：设计蓝图（讨论稿，2026-11-22 更新核心方法）· 配套对照页 `multiconformation_methods.html`
> 目标读者：ProFOLD2 多构象研究 · 本文记录目标、战略、核心难点、**核心方法（逐残基 switching 场 + 靶向条件化，取代全局 mode 加值）**、架构、评测与分阶段计划，以及被否决候选的决策依据。
> **核心原则**：feature + algorithm 必须显式刻画**哪些残基驱动构象变化**；全局均匀条件化不可行（探针 v1–v3 已证）。

## 1. 目标与战略定位

**目标**：无条件 `p(structure | sequence)` 多模采样——给一个序列（含**新序列**），采样出它的多个功能相关构象，核心指标是 **held-out 无条件 mode coverage**。

**战略**：
- **条件预测是 AF3 的强项**（它本就条件化配体/离子/修饰做预测）→ 在那竞争是冗余，**不做**。
- **无条件系综采样是 AF3 的短板**（mode collapse，双态恢复仅 ~20–23%，Peng 2025）→ 这才是真问题与价值点。
- 用户洞察："如果无条件采样能行，条件生成将会是一个简单任务"——条件生成是无条件能力的**免费推论**（从系综选 mode）。

**非目标（明确拒绝）**：
- 训练集 recall 原型 = 记忆/刷指标，不解决实际问题，**不做**。
- 纯条件预测 = AF3 已做，冗余。
- 侧链多构象 = 难定义、难从 PDB 挑选，**放弃**；聚焦**骨架级 fold-switching**。

## 2. 约束与资产

| 项 | 内容 |
|---|---|
| 条件信号 | 仅 target sequence，**无 MSA** |
| trunk | ProFOLD2 预训练编码器（Evoformer 148.74M / 98.2% 参数），冻结/轻调 |
| 解码器 | 自由重设计（保曲线约束只绑续训，不绑本研究） |
| 数据 | PDB 多态：同序列多条目（apo/holo 等），蛋白为主，**有条件标签**（配体/pH/伙伴） |
| 几何解码 | 复用 torsion-frame 机制（`rigids_from_angles` / `rigids_to_positions`） |

数据满足"分布学习必须见多态"的硬前提；条件标签是相对 AF3 的真正优势（AF3 训练集多态不足）。注意 PDB 多态 ≠ 完整 Boltzmann 系综（偏结晶条件、有限态、静态快照），足以学"采样功能态"，是务实基础。

## 3. 核心难点：无条件 mode coverage 不塌缩

难不在"会不会生成结构"，在"**能否把所有态采出来、不塌到主导态**"。三个子问题：

1. **trunk 表示平均**：预训练 trunk 按单 fold 训，repr 偏主导态；无条件多模需要 repr 编码序列的 **fold repertoire**（可能态集合），而非单一平均 fold。
2. **生成式解码器塌缩**：FM/diffusion 即使训了多态数据，也可能忽略噪声、只吐主导态。
3. **采样覆盖**：分布学到了，采样仍可能漏 mode。

**关键认知**：FM/diffusion 是 **commodity**（FM 行 → AF3 diffusion 也行）。AF3 有 diffusion 仍塌缩，因为杠杆**不在解码器**，而在：
- **多态数据**（硬前提）；
- **态特异、不平均的表示**；
- **生成式不塌缩 + 采样覆盖**。

## 4. 条件标签的角色：CFG 训练脚手架

虽然产品是无条件，条件标签在**训练时**帮表示/解码器**分离态**（监督 disambiguation），dropout 让模型也学边缘系综。**标签是学会"分离态"的辅助。无条件覆盖采样走"混合采样"（枚举/采 mode + 条件生成），而非直接采 FM 边缘（见 §4.5）。**

- 训练：条件 FM + **condition dropout**（classifier-free guidance）。以概率 p 把 condition 换 null，模型同时学 `p(struct|seq,cond)`、`p(struct|seq)` 与 `p(cond|seq)`。
- 推理：给条件 → 条件生成（guidance 选 mode）；**无条件 → 混合采样**（采/枚举 mode + 条件生成，§4.5），不直接采 FM 边缘（会塌）。
- **一个模型，两种能力**；条件生成是免费推论。
- 推论：若条件完全决定态，每个 `(seq,cond)` 单 GT → 无均值塌缩 → 确定性解码器 + FAPE 就够；FM 只处理"条件不能确定的剩余多模"。

### 4.1 核心方法：逐残基 switching 场 + 靶向条件化（取代全局 mode 加值）

**核心设计原则**：feature + algorithm 必须显式刻画**哪些残基驱动构象变化**。**全局均匀 mode 加值不可行**——不区分决定残基与响应残基，无的放矢，推动不了正确的局部触发（探针 v1–v3 已证实全局/均匀条件化失败，见 §4.4）。

**物理依据**：
- fold-switching 由**少数决定残基**驱动（Porter："几个关键残基决定胜出构象"）；大部分残基基本不动。
- **局部决定残基变化 → IPA 注意力耦合传播 → 全局 fold 重排**（local→global，StructureModule 已有此机制）。
- 抓住决定残基 = 抓住构象变化的**因果把手**——这是解决多构象的关键，而非全局 mode 旗标。

**特征 = 逐残基 switching 场**（残基间异质，非全局同值）：
- `s_i` = 残基 i 的 switching 倾向/变化量；switching 残基强信号，非 switching 残基 ≈0。
- 显式编码"**哪个残基变、变多少**"——核心表征要求。

**算法**：
- 从 `(seq, mode)` **预测逐残基 switching 场**（推理只有 seq+mode、无 fold1，场须预测而非给定）。
- **靶向条件化** switching 残基 → IPA 把局部触发传播成全局 fold。

**监督信号**：fold0↔fold1 的**逐残基差异**（正确局部叠合后，避开全局 Kabsch 假象，见 §4.4）→ switching 场 GT。

**核心学习问题（最难也最关键）**：从 `(seq, mode)` 预测 switching 场，须区分**决定残基（因）vs 响应残基（果）**——位移大 ≠ 因果驱动。这是"feature+algorithm 体现哪些残基影响构象变化"的落地。

**mode 角色降级**：mode（0=主导/1=非主导）仍是态索引，但条件化**经 switching 场逐残基施加**，非全局加值。

**数据组织（沿用）**：
- 多模信号锚定**同序列多结构**（PDB 多态资产）；家族聚合只作辅助，**不可拿家族间差异冒充序列内多模**。
- mode-rank 二态（0=主导/1=非主导合集），序列相对、数据赋予，绕开全局 fold 词表问题。
- 平衡采样覆盖导向（比例是 coverage-vs-布居旋钮，非固定 1:1）。

**泛化命门（诚实）**：switching 场须从 seq+mode 泛化预测；能否泛化取决于数据多样性 + 决定残基可学性，**held-out 序列评测**逼向泛化（非 recall）。

### 4.2 数据扩充：AFDB 相似转移（伪标签增广，真值锚 PDB）

整理多态数据困难，用 **AFDB 规模**扩充，机制是**按结构相似性转移 fold-switching 标注**：PDB 蛋白 X 已知 switch（有 fold0/fold1）→ AFDB 蛋白 Y 结构与 X 很像 → 推断 Y 为**候选 fold-switcher**。

**两个必须打问号的步骤**：
1. **相似 ≠ 必然 switch**：fold-switching 是序列的能量景观性质（两 fold 自由能近平衡），同源物结构像**未必都 switch**（一个突变可开启/关闭）。结构相似给的是**倾向性先验（candidate）**，非证明。
2. **转移的是标签，不是替代结构**：AFDB 只有 Y 的 fold0（预测主导态）。推断"Y 会 switch"**不等于拿到 Y 的 fold1**。要 fold1 须把 **X 的 fold1 当模板**同源模建到 Y → 假设"替代 fold 家族内保守"，仍是**近似/噪声**。

**定位（置信度分层）**：

| 来源 | 提供 | 置信度 |
|---|---|---|
| PDB 实验多态 | (seq, fold0, fold1) 真值 | 高（训练 + held-out 评测锚点） |
| AFDB 相似转移 | candidate fold-switcher 标签 + 模板 fold1 | 低（伪标签，按相似度过滤/降权） |

- **PDB 多态当高置信真值**；**AFDB 转移当低置信增广**（扩充候选与序列多样性），**不可当真值**，否则教错 mode。
- AFDB 在柔性区（多态区）pLDDT 低/误差大 → 伪标签噪声更高，需置信度过滤。
- 评测必须在**实验 fold-switcher**（held-out）上，不在 AFDB 伪标签上。
- 本质是 **fold-switching 倾向性预测**（Porter/Dishman 等有文献）：可做成前置筛选器，先筛 AFDB 候选再聚焦获取多态（模板/MD/实验），比盲目整理全 AFDB 高效。
- **加剧失衡**：AFDB 海量扩充 fold0，fold1 仍稀缺（仅 PDB/模板）→ 平衡采样比例需相应调整。

**替代 fold 保守性证据（模板转移的依据与边界）**：
- Porter 综述（Curr Opin Struct Biol 2024）：metamorphic 蛋白通常**源于单折叠祖先、在亚家族内保守**（RfaH 同序列分支常共享 switching；MinE/Mad2 近乎普遍保守）。
- Schafer & Porter（Nat Commun 2023, **ACE**）：超家族 MSA 剪成**亚家族特异 MSA** 才暴露替代 fold 共进化信号（56/56 switcher 显著）→ 替代 fold **保守但信号弱、亚家族特异**。
- Chakravarty（Nat Commun 2024）：AF 找替代 fold **主要靠结构记忆/序列关联**，成功集中在**高同源（~40–70%+ identity）**；<35% identity 基本失败（BCCIPα/β 80% 相同却不同 fold 为反例）。
- → **模板转移置信度按序列 identity 分层**：高同源主链模板可信，低同源替代 fold 未必保守 → 降权/丢弃。与"backbone-only fold1 + 置信度分层"吻合。

### 4.3 fold1 表示：backbone-only + frame 级生成 + 侧链后补

**fold1 常只有主链结构**（替代态多低分辨率/瞬态：cryo-EM 柔性态、NMR 系综、预测替代 fold、结构域模板）。这**不是缺陷而是契合**：fold-switching 本质是**骨架拓扑变化**，backbone 即多模核心，侧链次要。

- **生成任务统一在 backbone frame 级**：fold0/fold1 都生成骨架（frame FM 天然适配），mode_rank 控制哪个 fold；**侧链堆积作单独确定性步骤**（用 fold0 全原子数据训现有 torsion 头）。多模生成只管骨架，任务更纯更易学。
- **训练不对称**（ProFOLD2 loss 本分离 backbone/sidechain FAPE）：fold0 全原子 FAPE；**fold1 只算 backbone FAPE，mask 侧链 loss**。
- **放宽数据**：backbone trace/模板即可用 → AFDB 转移/NMR/cryo-EM 主链系综都可用。
- **评测对齐**：mode coverage 用 **backbone TM-score/RMSD**（fold-switching 本为骨架级），不要求 fold1 全原子。
- **caveat**：fold1 侧链重排（核心 repacking）学不到，对 fold-switching 检测/召回次要，可后补堆积。

### 4.4 探针教训与转向：全局条件化失败 → 逐残基 switching 场

**探针 v1–v3 实测（XCL1 单序列双 fold，400 步）**：
- v1 加性 mode_emb（trunk 输入）：被 evoformer pre-norm LayerNorm 洗掉（正常尺度存活 <6%）。
- v2 FiLM（to_single_repr 输出）：又被 folding head `single_repr_norm` 洗掉（注入点仍在 norm 前）。
- v3 条件 LayerNorm（folding head 4 norm，全微调 lr1e-5）：最佳 fold1 偏向（mode1→fold1 3.20 < mode1→fold0 3.71），但**远未记忆**（应~0）。
- **结论修正**：失败主因是**注入点 bug（norm 前被洗）+ 训练预算不足 + 单案例**，**不是**"trunk 表示不了 fold1"。早期"fold0 先验障碍/表示受限"结论**撤回**（过度解读）。

**关键教训**：
- ProFOLD2 LayerNorm 极多（trunk 700+，folding head 每 IPA block attn/ff norm）；**任何 norm 前的加性注入都被洗**。条件化须 post-norm / 条件 LN / extra token（attention 传播）。
- **全局均匀条件化不可行**：不区分决定/响应残基，无的放矢。
- 148M 容量本该轻松记忆双 fold；停在 3 Å 是注入+训练问题，非容量。

**转向逐残基 switching 场（§4.1 核心）**：
- 不再全局 mode 加值；改为**预测逐残基 switching 场 + 靶向条件化决定残基 + IPA local→global**。
- **align caveat**：XCL1 逐残基位移在全局 Kabsch 下弥散（95%>1Å），但中段核心稳定（~1Å）——弥散部分是叠合假象；switching 场须用**局部/核心叠合**识别，避开全局 Kabsch。

**修订探针（测核心假设）**：
- 测**模型能否预测逐残基 switching 场并据此驱动正确 fold 变化**，而非全局 mode 能否产 fold1。
- 监督：fold0↔fold1 逐残基差异（局部叠合）；多案例 + 足够训练预算 + held-out。
- 隔离 confound：注入 post-norm、训练充分、多案例、正确叠合。

**诚实风险**：seq-only + 数据稀缺仍不利；决定残基识别（因 vs 果）是核心难点；先验证 switching 场可学性，再搭全管线。

### 4.5 无条件采样 = 混合采样（mode + 条件生成），非直接 FM 边缘

**张力**：首要目标是无条件多模采样，但连续 FM 在多模上会平均塌缩（§5.1）。
- **条件生成**（mode_rank=0/1）：单模 → FM 无模态可平均 → 安全。
- **直接采 FM 边缘**（drop mode_rank 采 `p(struct|seq)`）：边缘多模 → 高噪声条件均值**又平均两 fold** → 塌缩风险回来。**首要目标恰落在 FM 最不稳处。**

**解法：分解无条件分布，混合采样**：
\[ p(\text{struct}\mid\text{seq}) = \sum_{m\in\{0,1\}} p(m\mid\text{seq})\, p(\text{struct}\mid\text{seq}, m) \]
1. 采 mode `m ~ p(m|seq)`（categorical，**无平均**）；
2. 条件生成 `struct ~ p(struct|seq, m)`（**单模 FM，无平均**）。

→ 多模放在 **categorical mode 选择**（安全），连续 FM 只做**单模条件生成**（安全）。**这是离散混合优势落到采样策略，mode 用数据标签 mode_rank，不需 VQ。**

**"无条件"的含义**：不给定具体 mode、生成整个系综——**枚举/采样 mode_rank 再条件生成**（扫 mode_rank=0,1 或按 `p(m|seq)` 采）。**mode coverage 由构造保证**（扫遍 mode 必采各态），不靠 FM 边缘"碰巧"多模。对新序列仍是泛化（从序列生成 fold，非检索），mode_rank 只选生成哪个 fold。

**CFG 角色（修正）**：dropout 学条件 + 边缘 + `p(m|seq)`；但**覆盖采样走混合路径**，非 drop 条件直接采 FM 边缘。`p(m|seq)` 可学 categorical 头，或覆盖导向均衡枚举两 mode。

**边界**：仍依赖条件生成质量（§4.4 探针）与序列→fold 泛化。若 fold1 内部多模（非主导合集 >2 fold），fold1 分支才需 VQ/进一步分解；二态下 fold1 通常单替代，问题不大。

### 4.6 非 switcher 幻觉：倾向门 + 置信度缓解

**问题**：混合采样枚举 mode_rank=0,1 默认"每序列都有替代态"，但**绝大多数蛋白不 switch**。对非 switcher 查询 mode_rank=1 是 **OOD 输入**（fold1 FM 只在 switcher 上训）→ 生成**幻觉替代 fold**。

**缓解 1：switching 倾向门 `p(m=1|seq)`**（fold-switching 预测，Porter/Dishman）：
- 非 switcher `p(m=1|seq)≈0` → 只生成 fold0；switcher 才生成 fold1。
- **scope 区分**：held-out **已知 switcher** 评测可省门（已知会 switch）；**通用部署（任意序列）必需**，否则幻觉淹没。
- 训练：正例（switcher）稀缺来自 PDB；**负例（非 switcher）由 AFDB 单 fold 海量提供**（AFDB 另一正当用途）。类别极不平衡 → 需校准，门质量是系统 precision 上限。

**缓解 2：置信度过滤（pLDDT/pTM）**：幻觉/OOD 结构通常**置信度低** → 低 pLDDT/pTM 可作软门过滤明显垃圾，降低幻觉危害。
- **关键反例（不能只靠置信度）**：Chakravarty 2024 表明 **AF 置信度反而"歧视"真实替代 fold**（替代态置信度偏低）；fold-switcher 本就比单折叠蛋白更不稳定/异质（Chakravarty 2023）。→ **真实替代 fold 与幻觉都偏低置信度**，单靠阈值无法干净区分，**过滤会误删真实替代态**。
- → 置信度是**有用的软缓解**（滤明显垃圾），但**不是倾向门的干净替代**；两者互补。部署用"倾向门 + 置信度"联合，评测已知 switcher 时置信度仅作辅助诊断。

**候选实现：生成模型兼任 fold-switching 分类器（倾向门由此导出，而非另训）**：
- 查询 mode_rank=0/1，比较 fold0 与 fold1：**分类信号 = 结构差异度 × 合理性**。
  - switcher：fold1 与 fold0 **结构不同**（fold0↔fold1 TM-score 低）**且合理**（几何/物理 plausible）。
  - 非 switcher：fold1 **塌回 fold0**（TM-score 高）或**幻觉 distinct 垃圾**（差异高但不合理）。
- **差异度比置信度干净**：真实替代 fold 置信度虽低，但与 fold0 确实不同 → 差异度高，避开"替代态低置信"偏置。但差异度 alone 不够（幻觉也 distinct）→ 须配**合理性判别**（立体化学/物理能量/学习判别器）滤垃圾。
- **统一**：`p(switch|seq)` 从 mode_rank=1 输出质量导出（distinct + plausible → 高），生成 + 分类一个模型。先例：AF2 MSA 子采样检测 switching、ACE 共进化信号——本方案是其生成式版本。
- **caveat**：依赖生成能力（§4.4 探针先行，循环依赖）；非 switcher 的 fold1 塌回 vs 幻觉需实测；合理性判别器需训/验证；差异度×合理性→p(switch) 需校准。属候选思路，待验证。

**小结**：幻觉问题真实但可控——倾向门（部署必需）+ 置信度软过滤（互补，注意真实替代态也低置信）；AFDB 供负例。倾向门可由"生成模型兼任分类器（差异度×合理性）"实现，生成/分类统一。当前 held-out switcher scope 可后置门，但通用化必须补上。

## 5. 候选机制（mode 如何表示/采样）

| 机制 | 如何保 mode 分离 | 优势 | 风险 |
|---|---|---|---|
| **连续 FM + 数据 mode-rank 条件 + CFG（基线）** | 数据标签 (0=主导/1=非主导) 监督分离 + dropout 学边缘 | 最简单、mode 由数据给定无需 discover、几何精确、条件/无条件一体 | 泛化靠序列→布居秩映射；连续去噪高噪声平均倾向 |
| 学习离散 mode latent（VQ）+ 连续精修（无标签备选） | 离散 code 天然分离 mode | 无监督 discover mode、离散保多模、或更省样本 | 非保距、codebook collapse、两阶段复杂度 |
| 能量头 + PT 采样 | 能量景观 + 跨 basin 采样 | 采稀有态、物理可解释 | 采样慢、能量难训 |

**基线：数据构造 mode-rank 标签（0=主导/1=非主导）+ 连续 FM + CFG（见 §4.1）。备选：VQ 离散混合——仅在无 mode 标签、需无监督发现时用。**

定位澄清（重要）：**数据 + 态特异 repr 是主杠杆（必要，与解码器无关）；解码器是次要架构选择**。有数据 mode 标签时，条件 FM + CFG 最简且直接；VQ 的"省样本/保分离"优势是**假设**，且其无监督 discover 在有标签时**不必要**。若基线 mode coverage 不足，再考虑 VQ 或采样增强，实测决定。

### 5.1 塌缩根源分析：连续去噪的算法弱点 + 离散混合的假设性优势

> 回答"为何 FM/diffusion 无条件采样多构象不理想，根源是不是算法本身"。AF3 以序列(+配体)为条件，无配体时≈ `p(struct|seq)`（即本文"无条件"），它确实塌缩——问题真实。

**核心算法机制：去噪/流匹配 = 条件均值回归 → 高噪声处平均模态。**
- Diffusion 训练是 MSE 回归噪声，最优解 `ε_θ(x_t,t,c) = E[ε | x_t, c]`（条件期望）；FM 同理 `v_θ = E[x_1−x_0 | x_t, c]`。
- 高噪声/中间时刻 `x_t` 近纯噪声，条件期望**把所有模态平均** → score/velocity 指向**平均结构**。
- 多模性只能在反向过程**末段（低噪声）**分化；若分化太晚或模态间 score 没学好，样本塌到主导/平均态。
- → 连续去噪的"条件均值回归"本性会在高噪声处平均多模，多模恢复是脆弱的末段行为。**这是 diffusion/FM 共有的算法倾向。**

**叠加的两个算法倾向**：① 反向动力学偏向高密度模态（主导 fold basin 更深，稀有态被丢）；② 模态间低密度区 score 估计误差（有限样本下分叉学不准）。

**但不是根本不可能**：diffusion/FM 是似然基（KL(data‖model) **mode-covering**，惩罚漏模态，不像 GAN mode-seeking），且是分布万能逼近器，**无表达力壁垒**。塌缩是 practical（估计/动力学/数据）问题，非 theoretical 不可能。

**主因其实是数据 + 条件化（可修复、占主导）**：trunk repr 按单 fold 训→对 fold-switcher 平均；PDB 多单态→学到的 `p(struct|seq)` 本就单峰。**杠杆在 repr + 数据，不在换解码器算法。**

**离散混合的假设性优势（待实测，非定论）**：
> 连续去噪会在高噪声处平均多模（条件均值本性）；离散混合把多模性**从连续去噪挪进 categorical `p(z|repr)`**（不会平均），让每个连续分量 `p(frames|repr,z)` **单模**——单模去噪没有模态可平均。

机制上离散混合**正面规避"条件均值平均多模"**，而非靠"离散看起来分离"。**但这是假设，不是证明**：该算法倾向在**数据稀缺时被放大**（模态间 score 估不准），故 VQ 的离散分解可能在少样本多模 regime 更省样本（每个子问题更简单）；然而数据充足时连续 FM 的似然 mode-covering 可能已足够，VQ 未必更优。**VQ 是否真比连续 FM 基线好，必须实测，不能先验断言。**

| 根源 | 性质 | 可修复性 |
|---|---|---|
| 条件均值回归→高噪声平均多模 | **算法本性**（diffusion/FM 共有） | 离散混合规避 / 引导缓解 |
| 反向动力学偏主导模态 | 算法倾向 | 增强采样/温度/repulsive |
| 模态间 score 估计误差 | 统计（有限数据） | 更多数据/引导 |
| trunk repr 平均 | **条件化**（主因） | 态特异 repr / 条件早注入 |
| 数据单态为主 | **数据**（主因） | 多态数据（本组有） |

**结论**：不是 diffusion 根本不行，是"条件均值平均多模"算法倾向 + 平均 repr + 单态数据三者叠加。**多态数据 + 态特异 repr 是必做的主杠杆**（解决两条主因）；解码器层面，连续 FM + CFG 是基线，VQ 离散混合是"规避算法倾向"的**可选候选**——是否必要由实测决定。纯换 FM/diffusion 而不动 repr/数据无用；但动了 repr/数据后，连续 FM 基线可能已足够，VQ 须对照验证。

## 6. 架构（逐残基 switching 场 + 靶向条件化）

**核心架构**：

```
sequence ─► 预训练 trunk(轻调) ─► repr ─────────────────────┐
                                                            ▼
(seq, mode) ─► switching 场预测器 ─► 逐残基 switching 场 s_i ─► 靶向条件化(post-norm/条件LN/extra token)
                                                            │
                                                            ▼
                                            StructureModule IPA (local→global 传播)
                                                            │
                                                            ▼
                                                frames ─►(侧链后补)─► atom
```

- **switching 场预测器**：从 `(seq, mode)` 预测逐残基 switching 场 `s_i`（哪些残基变、变多少）；监督来自 fold0↔fold1 逐残基差异（**局部/核心叠合**，§4.4）。
- **靶向条件化**：`s_i` 施加到对应残基（switching 残基强、其余≈0）；**post-norm / 条件 LN / extra token**，避开 norm 前注入被洗（§4.4 教训）。
- **IPA local→global**：靶向扰动决定残基 → IPA 注意力耦合传播成全局 fold 重排。
- **生成统一 backbone frame 级**（§4.3）；侧链后补；fold1 只算 backbone FAPE、mask 侧链。
- 推理：mode=0/1 → 指定态 switching 场；**无条件 → 混合采样**（§4.5）。

**与旧全局 mode 架构的本质区别**：条件化**逐残基异质**（switching 场），非全局同值；**靶向决定残基**，非弥散广播。这是 §4.1 核心原则的架构落地。

**备选（VQ 离散 mode，仅无 switching 场监督时）**：VQ discover mode 码本 + 连续几何精修；但 switching 场有显式逐残基监督时优先（更直接表征"哪个残基变"）。

**先验证 switching 场可学性**（§4.4 修订探针：多案例 + 充分训练 + 正确叠合 + post-norm 注入），再搭全管线。

## 7. 评测协议（真指标）

- **按序列 split** train/val/test，test 序列训练里没见过（泛化命门）。
- **无条件混合采样**（§4.5：枚举/采 mode_rank + 条件生成）N 个 → 聚类（**backbone TM-score/RMSD**，fold-switching 为骨架级；fold1 backbone-only 不要求全原子）→ 匹配真实态 → **mode coverage（recall/precision）**。
- 对照：AF3 / 单态基线 / 确定性解码器。
- 辅助指标：条件精度（免费推论，非主指标）。
- **不是**训练 recall，**不是**条件精度。

## 8. 分阶段计划

- **Phase 0 · 数据组织 + mode 可分性**：
  - 多态 grouping **锚定同序列多结构**（跨条目）；**家族聚合辅助**识别 repertoire/标注，**严禁拿家族间差异冒充序列内多模**。
  - 构象聚类（TM-score/RMSD）→ **二态 mode-rank 标签：0=主导、1=非主导（合并替代）**；条件解析；**平衡采样**（覆盖导向，比例可调）。
  - **AFDB 相似转移增广**（§4.2）：按结构相似性从 PDB fold-switcher 转移候选标签 + 模板 fold1，**伪标签按相似度过滤/降权**，真值锚 PDB。
  - **按序列 split** + 统计（多态序列数、态数、态间 TM-score 分布——区分大 fold-switch vs 小波动）。**先验证 mode 可分性**。
- **Phase 0.5 · 根基探针（先于全管线，隔离 trunk 可表示性，按 identity 分层）**：repr 判别探针（mode_rank=0/1 是否产态特异 repr）+ **极简/冻结解码器**重建 fold1（避免解码器混淆，§4.4），held-out fold-switcher **按序列 identity 分层**测 TM-score（§4.4/§11）：高同源（>40%）能重建 → Tier 1 成立；低同源（<35%）失败 → 确认 Tier 3 不可达。**画"泛化 vs identity"曲线定能力边界**，而非单一 pass/fail。生死线，失败则重议编码器/scope。**设计已完备，此探针是唯一未验证根基——先测再定后续，不先验堆组件。**
- **Phase 1 · 路线 C 条件 FM 基线 pilot**：trunk 轻调 + mode_rank 注入 trunk **且 skip 到 frame FM 解码器**（AF3 式）+ CFG dropout + 平衡采样（fold0 确定性、fold1 生成式），小规模验证 **held-out 无条件 mode coverage**；对照确定性解码器（应单态）vs 条件生成式（应多态）。**先确认基线能否不塌缩且泛化。**
- **Phase 1b · VQ 备选消融**：仅当无 mode 标签或基线 mode coverage 不足时，上 VQ 离散混合（同 trunk/repr/评测），实测是否更优；有标签且基线足够则 VQ 不必要。
- **Phase 2 · 扩展**：采样增强（PT/repulsive）、条件化精修、trunk 进一步微调。
- **Phase 3 · 泛化评测**：held-out 无条件 mode coverage + 基线对比。

## 9. 诚实边界与风险

- **核心生死线：trunk 替代-fold 可表示性**（§4.4）。预训练 trunk 塌到主导 fold；seq-only 缺共进化信号（ACE 表替代 fold 信号在亚家族 MSA）。纯序列 trunk 能否编码替代 fold 未验证——**最大不确定性在此，不在解码器**。先做根基探针，失败则重议编码器/scope。
- **未解前沿**，无保证成功的配方（AF3 有 diffusion + 海量数据仍塌缩）。
- **数据稀缺是硬约束**：fold-switcher 少，mode 覆盖上限由数据多样性决定。
- **解码器是次要消融，非主杠杆**：数据 + 态特异 repr 必做；VQ 相对连续 FM 的"省样本/保分离"优势是假设，未证明，须基线对照实测——不先验押注 VQ。
- **mode-rank 条件的泛化难点**：标签只说"主导/替代"，没说是什么 fold；新序列须模型自学"序列→布居秩→fold"映射，非给条件即自动泛化。家族聚合须锚定同序列多结构，勿把家族间差异当序列内多模。
- **AFDB 伪标签风险**：相似 ≠ 必然 switch（倾向性假设）；转移的是标签非替代结构（模板 fold1 假设家族保守，噪声）；AFDB 柔性区 pLDDT 低。须降权/过滤，真值锚 PDB，评测在实验 fold-switcher 上；AFDB 扩充加剧 fold0/fold1 失衡。
- 离散 latent 风险：mode 发现质量、token→结构非保距（MultiStateFold：88.7% token 差异仍可 TM 0.96）、codebook collapse——需实测。
- 泛化到新序列的 fold repertoire 依赖数据多样性，非解码器/条件化能单独解决。
- **fold1 侧链缺失**：backbone-only fold1 学不到替代态侧链重排（核心 repacking），对 fold-switching 检测/召回次要，可后补堆积；模板转移置信度按序列 identity 衰减（高同源可信、<35% 基本失败）。

## 10. 被否决候选（决策记录）

| 候选 | 否决理由 |
|---|---|
| 训练集 recall（mode-index 记忆） | 刷指标，不解决实际问题 |
| **全局均匀 mode 加值/全局 mode token** | 不区分决定/响应残基，无的放矢；探针 v1–v3 失败（norm 前注入被洗 + 弥散无靶向）；改用逐残基 switching 场（§4.1） |
| 全局预定义 fold token（候选 3） | 跨蛋白 fold0/fold1 无法定义；退化为序列相对索引 = recall-only |
| 侧链多构象 | 难定义、难从 PDB 挑选 |
| 纯条件预测 | AF3 强项，冗余 |
| tied vs untied 结构模块 | 暂停；精度差未知，需从头消融，与续训不兼容 |

## 11. 能力边界与 scope 分层（战略 reality check）

**根本瓶颈是数据/信号，不是架构**——再加组件也突破不了泛化上限：
- 已知 fold-switcher **百量级**（Porter 2018 ~96–100），替代 fold 拓扑高度多样，样本覆盖不了"序列→替代 fold"映射的多样性。
- **seq-only 缺共进化信号**（ACE：替代 fold 信号在亚家族 MSA），比 AF2 带 MSA 更难。
- **Chakravarty 2024 警示**：AF2 带 MSA + 海量数据，找替代 fold 仍靠**高同源记忆**（~40–70%+），<35% 基本失败。**连 AF2 都泛化不了低同源 fold-switching**，seq-only + 百例更难。

**scope 三层（务实定位，不承诺做不到的）**：

| 层 | 目标 | 可行性 |
|---|---|---|
| Tier 1 | 高同源 fold-switching：为已知 switcher 的 close homolog 生成替代 fold | 较可行（高同源记忆区） |
| Tier 2 | fold-switching 检测/分类（§4.6 分类器 + AFDB 负例） | 较可行（分类比生成易，负例充足） |
| Tier 3 | 新低同源 fold-switching 预测 | 很难，当前数据下可能不可达 |

→ **先做 Tier 1 + Tier 2，Tier 3 当远期/数据充足后目标，不承诺。**

**identity 分层探针定边界**（§8 Phase 0.5）：按序列 identity 分层测 fold1 重建，画"泛化 vs identity"曲线——高同源成立确认 Tier 1，低同源失败确认 Tier 3 不可达。比单一 pass/fail 更有信息。

**seq-only niche 之问（novelty，须厘清）**：相对 Porter ACE / AF2 MSA 子采样（都需 MSA），本方案差异是 seq-only——
- 劣势：信号更少，精度可能不如 MSA 方法。
- 潜在 niche：**无 MSA 序列**（孤儿序列、设计蛋白、快速推理）——MSA 方法失效处。
- 须诚实回答：seq-only fold-switching 生成/检测相对 MSA 方法有何**不可替代**贡献？若只是"更弱但免 MSA"价值有限；若覆盖 MSA 做不到场景（如设计蛋白 fold-switching）才有意义。

**战略选择**：若目标是 Tier 3（真泛化），需**扩充多态数据**（MD 系综、cryo-EM 异质性、更多实验 switcher）或**引入共进化信号**（违背 seq-only？）——这是战略选择，非设计细节。当前设定下建议先 identity 分层探针定边界 + 定 scope（Tier 1+2）+ 想清 seq-only niche，再决定是否按此推进。

## 12. 核酸多构象数据 landscape（局部 vs 全局）

核酸多构象**同样数据受限**，但**不是没数据**——有现成 curated 数据集（文献调研，非亲测）：

- **SMARTFlexDB**（NAR/bioRxiv 2025/2026）：curated RNA **apo–holo 配对**库——**452 PDB 结构**、734 条目、266 holo/186 apo、**1187 pocket 级配对**、150 小分子、23 功能类；**riboswitch 占 38%（最大类）**；**自带 leakage-controlled splits**，为 benchmark 设计。
- **CoDNaS-RNA**（NAR 2022）：RNA 天然态构象多样性库；**RNA-Puzzles**：riboswitch aptamer 盲测靶标（含 bound/unbound）；**Ribocentre** riboswitch catalog。

**关键 nuance：核酸多态数据多是局部小变化，非大 fold-switching**：
- SMARTFlexDB 1187 配对统计：**全局 RMSD 中位 1.16 Å**、**TM-score 中位 0.79**、**碱基对重排中位 0** → 多数 apo–holo 差异是**局部 pocket 重排**。
- 有的 riboswitch（SAM-I）游离/结合态 aptamer **无显著构象变化**（Liberman & Penedo 2011）→ "riboswitch=fold-switching" 不总成立。

**与蛋白数据性质不同**：
- 蛋白 fold-switcher：少（百例）但**大拓扑切换**。
- 核酸 apo–holo：较多（452 结构/1187 对）但**局部小变化**——类似本文为蛋白**否决的"侧链/局部多构象"**那类。

**scope 决定可行性**：
- **局部柔性**（pocket/loop 重排）：核酸数据**反而更充足、更 curated**（SMARTFlexDB 是现成 benchmark 起点）。
- **大 RNA fold-switching**（表达平台重排等）：数据**仍稀缺**，与蛋白 fold-switching 同困境。
- → 先定**局部 vs 全局** scope 再判断数据够不够；"RNA 自由度大"属实，但可用多态数据捕获的多是局部柔性。

## 附：关键文献锚点

- AlphaFlow / ESMFlow：预训练 trunk + frame flow matching 系综（MD 训练）。
- BioEmu / DiG：Boltzmann 平衡分布学习。
- ConfDiff：力引导 SE(3) 扩散。
- MultiStateFold：ESM3 离散 token + 并行回火 MCMC（非保距证据）。
- Peng 2025（Briefings in Bioinformatics）：AF3 多态恢复 ~20–23%。
- ConforNets：latent 调制 eliciting 多态（fold-switcher 基准）。
- Porter 2024（Curr Opin Struct Biol, PMC11102287）：metamorphic 蛋白源于单折叠祖先、亚家族内保守。
- Schafer & Porter 2023（Nat Commun, ACE）：亚家族特异 MSA 暴露替代 fold 共进化信号（56/56）。
- Chakravarty 2024（Nat Commun, PMC11344769）：AF 替代 fold 靠结构记忆/序列关联，高同源（~40–70%+）成功、<35% 失败。
- SMARTFlexDB（NAR/bioRxiv 2025/2026）：RNA apo–holo 配对库，452 PDB/1187 对，riboswitch 38%，变化多局部（RMSD 中位 1.16 Å/TM 0.79）。
- CoDNaS-RNA（NAR 2022）：RNA 天然态构象多样性库。
- Liberman & Penedo 2011（PMC3252462）：SAM-I riboswitch 游离/结合态 aptamer 无显著构象变化。