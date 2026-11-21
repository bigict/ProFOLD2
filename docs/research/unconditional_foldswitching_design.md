# 无条件 Fold-Switching 构象系综采样 — 设计方案

> 状态：设计蓝图（讨论稿，2026-11-21）· 配套对照页 `multiconformation_methods.html`
> 目标读者：ProFOLD2 多构象研究 · 本文记录目标、战略、核心难点、基线（数据 mode-rank 条件 FM + CFG）与 VQ 无标签备选、架构、评测与分阶段计划，以及被否决候选的决策依据。

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

虽然产品是无条件，条件标签在**训练时**帮表示/解码器**分离态**（监督 disambiguation），dropout 让模型也学边缘系综；推理 drop 条件 → 无条件采样。**标签是学会"分离态"的辅助，不是推理输入。**

- 训练：条件 FM + **condition dropout**（classifier-free guidance）。以概率 p 把 condition 换 null，模型同时学 `p(struct|seq,cond)` 与 `p(struct|seq)`。
- 推理：给条件 → 条件生成（guidance 选 mode）；drop 条件 → 无条件采样。
- **一个模型，两种能力**；条件生成是免费推论。
- 推论：若条件完全决定态，每个 `(seq,cond)` 单 GT → 无均值塌缩 → 确定性解码器 + FAPE 就够；FM 只处理"条件不能确定的剩余多模"。

### 4.1 基线核心：数据构造 mode-rank 标签 + 条件 FM + CFG

用**数据构造的 mode-rank 标签**作为 CFG 的条件，比 VQ 无监督发现 mode 更简单——**这是基线首选**。

**数据组织**：
- **多模信号锚定"同序列多结构"**（本组 PDB 多态资产：同序列多条目）。**家族聚合只作辅助**——识别 fold repertoire、协助标注/扩充；**不可拿家族间不同序列的结构差异冒充序列内多模**（那是普通序列→fold 预测，不是多构象）。
- **mode-rank 标签（二态：主导/非主导）**：**0 = 主导态，1 = 非主导态（合并所有替代 fold）**。含义**序列相对、由数据赋予**，不是全局 fold 词表 → 绕开"fold0/fold1 跨蛋白无法定义"的否决理由。
  - **为何二态而非 0/1/2…k**：k-秩排序对多态不稳定、难定义；合并 2..k 到"非主导"避开它。
  - **代价（关键）**：fold0 单模（易，确定性解码器即可收敛）；**fold1 = 非主导合集，本身多模** → 多模被**隔离进 fold1 分支**，该分支需生成式解码器。这是干净的不对称分解：大部分预测只需 fold0，少数需 fold1（采替代系综）。
- **平衡采样**：主导/非主导按比例见样，**覆盖导向**（稀有态充分学到），非物理布居导向；比例是 coverage-vs-布居旋钮，非固定 1:1（AFDB 扩充会加剧 fold0/fold1 失衡，见 §4.2）。

**模型/训练**：
- 条件生成 `p(struct | seq, mode_rank)`：**fold0 单 GT** → 无均值塌缩，确定性 + FAPE 即可；**fold1 多 GT（非主导合集）** → 仍需生成式（FM）建模其多模。条件化把多模**限制在 fold1 分支**，fold0 易收敛。
- **CFG dropout**：训练随机 drop mode_rank → 也学边缘 `p(struct|seq)`；推理 drop → 无条件采样。**条件/无条件一个模型**。
- **mode-rank 查询是 AF3 没有的能力**：显式"给主导/替代 fold"，AF3 只吐单态 → 差异点，非冗余。

**为何比 VQ 简单**：mode 由数据标签给定，无需无监督 discover（避开 codebook collapse / 非保距 / 两阶段）。**VQ 退为"无 mode 标签、需无监督发现"时的备选。**

**泛化命门（诚实）**：mode-rank 只说"主导/替代"，**没说是什么 fold**；对新序列模型须自学"序列→布居秩→fold"映射。能否泛化取决于数据多样性 + 序列决定因素可学性，**不是给条件就自动解决**。**held-out 序列评测**把它逼向泛化（架构同 mode-conditioned，但目标/评测是泛化而非 recall；家族聚合增每 mode 样本助泛化）。

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

## 6. 架构（mode-rank 条件 FM 基线 + VQ 无标签备选）

**基线架构（先做）**：

```
sequence ─► 预训练 trunk ─► 态特异 repr ─► 连续 frame FM 解码器 ─► frames ─► torsion ─► atom
                  ▲                              ▲
            mode_rank 早注入                CFG mode_rank dropout
            (0=主导/1=非主导, 数据标签)        (训练脚手架, 推理 drop)
```

- trunk 冻结/轻调，**数据构造的 mode_rank（0=主导/1=非主导）早注入**产态特异 repr；连续 frame FM 学 `p(frames|repr, mode_rank)`，CFG dropout 学边缘 `p(frames|repr)`。
- 推理：mode_rank=0/1 → 指定态；drop → 无条件系综。
- 最简单，先验证"同序列多态数据 + mode-rank 条件 + 态特异 repr + CFG 能否让连续 FM 不塌缩、held-out 泛化"。

**备选架构（VQ 混合，仅无 mode 标签时）**：

```
sequence ─► 预训练 trunk ─► 态特异 repr ─► 离散 mode latent (VQ) ─► mode code
                  ▲                              │
            condition 早注入                     ▼
            (CFG dropout 脚手架)         mode code + repr
                                               │
                                               ▼
                                  连续几何解码器 (frame FM/flow)
                                               │
                                               ▼
                                     frames ─► torsion ─► atom
```

- **trunk**：条件早注入（借鉴 `conditional_pos` 注入点，但需语义条件编码而非几何 distogram），产态特异表示；冻结或轻调。
- **离散 mode latent**：VQ-VAE 在结构/表示空间 discover mode 码本；采样 code 即选 mode（离散、分离）。
- **连续几何精修**：mode code 条件化 frame flow matching，生成精确骨架 → 复用 torsion 解码到全原子。
- **训练**：CFG condition dropout（条件标签脚手架）+ 似然目标（不平均）。
- **两架构共享 trunk/repr/条件/评测**，仅解码器不同 → 干净消融，隔离"离散 vs 连续"这一个变量。

## 7. 评测协议（真指标）

- **按序列 split** train/val/test，test 序列训练里没见过（泛化命门）。
- **无条件**采样 N → 聚类（TM-score/RMSD）→ 匹配真实态 → **mode coverage（recall/precision）**。
- 对照：AF3 / 单态基线 / 确定性解码器。
- 辅助指标：条件精度（免费推论，非主指标）。
- **不是**训练 recall，**不是**条件精度。

## 8. 分阶段计划

- **Phase 0 · 数据组织 + mode 可分性**：
  - 多态 grouping **锚定同序列多结构**（跨条目）；**家族聚合辅助**识别 repertoire/标注，**严禁拿家族间差异冒充序列内多模**。
  - 构象聚类（TM-score/RMSD）→ **二态 mode-rank 标签：0=主导、1=非主导（合并替代）**；条件解析；**平衡采样**（覆盖导向，比例可调）。
  - **AFDB 相似转移增广**（§4.2）：按结构相似性从 PDB fold-switcher 转移候选标签 + 模板 fold1，**伪标签按相似度过滤/降权**，真值锚 PDB。
  - **按序列 split** + 统计（多态序列数、态数、态间 TM-score 分布——区分大 fold-switch vs 小波动）。**先验证 mode 可分性**。
- **Phase 1 · mode-rank 条件 FM 基线 pilot**：trunk 冻结 + mode_rank 早注入态特异 repr + 连续 frame FM（fold0 可确定性、fold1 生成式）+ CFG dropout + 平衡采样，小规模验证 **held-out 无条件 mode coverage**；对照确定性解码器（应单态）vs 条件生成式（应多态）。**先确认基线能否不塌缩且泛化。**
- **Phase 1b · VQ 备选消融**：仅当无 mode 标签或基线 mode coverage 不足时，上 VQ 离散混合（同 trunk/repr/评测），实测是否更优；有标签且基线足够则 VQ 不必要。
- **Phase 2 · 扩展**：trunk 轻调、采样增强（PT/repulsive）、条件化精修。
- **Phase 3 · 泛化评测**：held-out 无条件 mode coverage + 基线对比。

## 9. 诚实边界与风险

- **未解前沿**，无保证成功的配方（AF3 有 diffusion + 海量数据仍塌缩）。
- **数据稀缺是硬约束**：fold-switcher 少，mode 覆盖上限由数据多样性决定。
- **解码器是次要消融，非主杠杆**：数据 + 态特异 repr 必做；VQ 相对连续 FM 的"省样本/保分离"优势是假设，未证明，须基线对照实测——不先验押注 VQ。
- **mode-rank 条件的泛化难点**：标签只说"主导/替代"，没说是什么 fold；新序列须模型自学"序列→布居秩→fold"映射，非给条件即自动泛化。家族聚合须锚定同序列多结构，勿把家族间差异当序列内多模。
- **AFDB 伪标签风险**：相似 ≠ 必然 switch（倾向性假设）；转移的是标签非替代结构（模板 fold1 假设家族保守，噪声）；AFDB 柔性区 pLDDT 低。须降权/过滤，真值锚 PDB，评测在实验 fold-switcher 上；AFDB 扩充加剧 fold0/fold1 失衡。
- 离散 latent 风险：mode 发现质量、token→结构非保距（MultiStateFold：88.7% token 差异仍可 TM 0.96）、codebook collapse——需实测。
- 泛化到新序列的 fold repertoire 依赖数据多样性，非解码器/条件化能单独解决。

## 10. 被否决候选（决策记录）

| 候选 | 否决理由 |
|---|---|
| 训练集 recall（mode-index 记忆） | 刷指标，不解决实际问题 |
| 全局预定义 fold token（候选 3） | 跨蛋白 fold0/fold1 无法定义；退化为序列相对索引 = recall-only |
| 侧链多构象 | 难定义、难从 PDB 挑选 |
| 纯条件预测 | AF3 强项，冗余 |
| tied vs untied 结构模块 | 暂停；精度差未知，需从头消融，与续训不兼容 |

## 附：关键文献锚点

- AlphaFlow / ESMFlow：预训练 trunk + frame flow matching 系综（MD 训练）。
- BioEmu / DiG：Boltzmann 平衡分布学习。
- ConfDiff：力引导 SE(3) 扩散。
- MultiStateFold：ESM3 离散 token + 并行回火 MCMC（非保距证据）。
- Peng 2025（Briefings in Bioinformatics）：AF3 多态恢复 ~20–23%。
- ConforNets：latent 调制 eliciting 多态（fold-switcher 基准）。