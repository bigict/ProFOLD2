# 无条件 Fold-Switching 构象系综采样 — 设计方案

> 状态：设计蓝图（讨论稿，2026-11-21）· 配套对照页 `multiconformation_methods.html`
> 目标读者：ProFOLD2 多构象研究 · 本文记录目标、战略、核心难点、候选机制、主推架构、评测与分阶段计划，以及被否决候选的决策依据。

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

## 5. 候选机制（mode 如何表示/采样）

| 机制 | 如何保 mode 分离 | 优势 | 风险 |
|---|---|---|---|
| 连续 FM + CFG | 噪声索引 mode + 条件脚手架 | 成熟（AlphaFlow）、几何精确 | 易塌缩/插值混合态 |
| **学习离散 mode latent（VQ）+ 连续精修** | 离散 code 天然分离 mode | 离散保多模、**mode 从数据 discover** | 非保距、codebook collapse |
| 能量头 + PT 采样 | 能量景观 + 跨 basin 采样 | 采稀有态、物理可解释 | 采样慢、能量难训 |

**主推：学习离散 mode latent + 连续几何精修。**

关键连接：之前否决"fold0/fold1 跨蛋白无法预定义"——但 **VQ 的 mode 是从数据 discovered，不是预定义**，恰好绕开死结（不需要通用 fold 词表，模型自学 mode 码本）。离散 mode 天然不插值 → 对"覆盖分离的全局 fold"比连续 FM 更稳；几何精度交给后续连续精修。**离散保 mode 分离，连续保几何精度，各司其职。**

### 5.1 塌缩根源分析：连续去噪的算法弱点 + 离散混合的硬理由

> 回答"为何 FM/diffusion 无条件采样多构象不理想，根源是不是算法本身"。AF3 以序列(+配体)为条件，无配体时≈ `p(struct|seq)`（即本文"无条件"），它确实塌缩——问题真实。

**核心算法机制：去噪/流匹配 = 条件均值回归 → 高噪声处平均模态。**
- Diffusion 训练是 MSE 回归噪声，最优解 `ε_θ(x_t,t,c) = E[ε | x_t, c]`（条件期望）；FM 同理 `v_θ = E[x_1−x_0 | x_t, c]`。
- 高噪声/中间时刻 `x_t` 近纯噪声，条件期望**把所有模态平均** → score/velocity 指向**平均结构**。
- 多模性只能在反向过程**末段（低噪声）**分化；若分化太晚或模态间 score 没学好，样本塌到主导/平均态。
- → 连续去噪的"条件均值回归"本性会在高噪声处平均多模，多模恢复是脆弱的末段行为。**这是 diffusion/FM 共有的算法倾向。**

**叠加的两个算法倾向**：① 反向动力学偏向高密度模态（主导 fold basin 更深，稀有态被丢）；② 模态间低密度区 score 估计误差（有限样本下分叉学不准）。

**但不是根本不可能**：diffusion/FM 是似然基（KL(data‖model) **mode-covering**，惩罚漏模态，不像 GAN mode-seeking），且是分布万能逼近器，**无表达力壁垒**。塌缩是 practical（估计/动力学/数据）问题，非 theoretical 不可能。

**主因其实是数据 + 条件化（可修复、占主导）**：trunk repr 按单 fold 训→对 fold-switcher 平均；PDB 多单态→学到的 `p(struct|seq)` 本就单峰。**杠杆在 repr + 数据，不在换解码器算法。**

**离散混合的硬理由（正面规避算法根源）**：
> 连续去噪会在高噪声处平均多模（条件均值本性）；离散混合把多模性**从连续去噪挪进 categorical `p(z|repr)`**（不会平均），让每个连续分量 `p(frames|repr,z)` **单模**——单模去噪没有模态可平均。

即离散混合**正面规避"条件均值平均多模"这个算法根源**，而非靠"离散看起来分离"。这是它相对纯连续 FM 的 principled 优势，也回答"离散是否必要"——它针对连续去噪的真实弱点。

| 根源 | 性质 | 可修复性 |
|---|---|---|
| 条件均值回归→高噪声平均多模 | **算法本性**（diffusion/FM 共有） | 离散混合规避 / 引导缓解 |
| 反向动力学偏主导模态 | 算法倾向 | 增强采样/温度/repulsive |
| 模态间 score 估计误差 | 统计（有限数据） | 更多数据/引导 |
| trunk repr 平均 | **条件化**（主因） | 态特异 repr / 条件早注入 |
| 数据单态为主 | **数据**（主因） | 多态数据（本组有） |

**结论**：不是 diffusion 根本不行，是"条件均值平均多模"算法倾向 + 平均 repr + 单态数据三者叠加。多态数据解决一条、态特异 repr 解决一条、离散混合规避算法那条——**三管齐下**才完整；纯换 FM/diffusion 无用，必须同时动 repr/数据/模态表示。

## 6. 主推架构

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

## 7. 评测协议（真指标）

- **按序列 split** train/val/test，test 序列训练里没见过（泛化命门）。
- **无条件**采样 N → 聚类（TM-score/RMSD）→ 匹配真实态 → **mode coverage（recall/precision）**。
- 对照：AF3 / 单态基线 / 确定性解码器。
- 辅助指标：条件精度（免费推论，非主指标）。
- **不是**训练 recall，**不是**条件精度。

## 8. 分阶段计划

- **Phase 0 · 数据地基 + mode 可分性**：多态 grouping（按序列跨条目）+ 构象聚类 + 条件解析 + 按序列 split + 统计（多态序列数、态数、态间 TM-score 分布——区分大 fold-switch vs 小波动）。**先验证 mode 可分性**，决定离散 latent 可行性。
- **Phase 1 · 最小 pilot**：trunk 冻结 + 离散 mode latent + 连续精修 + CFG，小规模验证无条件 mode coverage；对照确定性解码器（应单态）vs 生成式（应多态）。
- **Phase 2 · 扩展**：trunk 轻调、采样增强（PT/repulsive）、条件化精修。
- **Phase 3 · 泛化评测**：held-out 无条件 mode coverage + 基线对比。

## 9. 诚实边界与风险

- **未解前沿**，无保证成功的配方（AF3 有 diffusion + 海量数据仍塌缩）。
- **数据稀缺是硬约束**：fold-switcher 少，mode 覆盖上限由数据多样性决定。
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