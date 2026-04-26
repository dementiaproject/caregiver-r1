# EC-MTRL：双时域奖励解耦下的多轮对话 RL —— 以痴呆护理事实冲突对话为 Testbed

**Environment-Coherent Multi-Turn RL: A Two-Horizon Reward-Decoupled Framework for Multi-Turn Dialogue RL with Strategy-Conditioned Rollouts, Validated on Safety-Critical Dementia Care Conversations**

---

> **Note (2026-04-26)** — this document is the long-form method draft frozen
> at zh_9. Three architectural decisions taken after this draft override parts
> of it; they are documented in `PROPOSAL.md §9` ("Architectural decisions
> that supersede zh_9 in places"). Code is the authoritative source. When you
> read this paper, mentally apply:
>
> - **Decision 1** — unified-prompt sampling (not strategy-conditioned);
> - **Decision 2** — single-layer LLM safety judge (not regex C_cat +
>   Lagrangian c_safety) — superseded by Decision 3;
> - **Decision 3** — binary hard-veto safety: `c_safety ∈ {0, 1}`, three
>   catastrophic-only items OR-aggregated, gate + fixed floor penalty
>   (`λ_violation ≈ 5.0`, NOT Lagrangian). Stylistic / empathy concerns
>   live in R_fit's negative-points criteria, not in c_safety.

---

## 摘要

多轮对话 RL 的核心瓶颈是**轮级 credit assignment**：一段 8–10 轮对话的最终质量好坏应归因到哪一轮决策？现有方法（MAPO、MT-GRPO、HCAPO、DuCA）都默认 LLM simulator 只输出文本，被迫为轮级信号另起一套外部机制——per-turn LLM-judge（贵）、折扣反推（强假设）、反事实推理（不稳）。我们的关键观察是：**RL 环境本身（DemMA simulator）在与 agent 交互时已经产出可被直接作为奖励的反馈信号**——当 simulator 是 LLM 时，这些信号是在生成 utterance 同一次 forward pass 中 commit 的离散 action labels。我们直接把这种 *environment-native feedback* 作为 RL reward，不再依赖外部训练的 reward model 或事后 LLM judge 推断；它满足三条可代码级审计的工程性质：inline commitment、zero marginal cost、mappable to validated clinical scales。把这些信号经临床量表（OERS / PAINAD / RTC）锚定的 ordinal 映射，即得到一条 non-anticipative、零额外 LLM 调用的轮级奖励通道。基于此，我们提出 **EC-MTRL**——一个 GDPO 风格的多轮变种：(i) 把奖励解耦为**双时域**（轨迹级 checklist judge + 轮级 inline 映射），各自独立归一化以避免 broadcast 主导；(ii) 把 rollout group 的来源从"同一 policy 采样 N 次"换成"**10 个临床护工策略各 prompt 一次**"，让 group 内带有 principled 行为多样性；(iii) 两层安全约束（规则硬否决 + judge Lagrangian），安全不进入可补偿奖励池。我们以痴呆护理事实冲突对话为主 testbed（DemMA 冻结环境），围绕 5 个研究问题（vs SFT/GRPO/PPO 经典 baseline，vs MAPO/MT-GRPO/RLVER 同类多轮方法，vs 人类护工 reference）做 pre-registered 评估，并以 3 层 reward-hacking audit 公开 report 失败模式。本工作定位为 **Use-Inspired** 的可行性研究：方法贡献是把环境同步产出的 inline annotation **重新框定**为多轮 RL 的 *environment-native reward signal*，testbed 贡献是为 ML 社区提供一个临床有据、安全张力真实的 dialog RL 评测场景。

---

## §1 引言

### 1.1 一个具体的临床场景

> *设定：晚期痴呆患者王女士，记忆停留在 1990 年代。下午 3 点，她在护理中心走廊上拦住一位护工：*
>
> **王女士**：我妈妈今天来看我了，她在楼下等我，我得下去。
> **护工 A**（直接纠正）：王阿姨，您妈妈已经过世 12 年了，她不会来看您。
> **护工 B**（顺其自然）：好，那您先在这儿坐一会儿，我去帮您看看。
> **护工 C**（情绪共情 + 转移）：您一定很想她吧？我们去活动室坐下，您跟我说说她以前做的菜好吗？

三种回应都"听起来合理"。在临床上：A（Reality Orientation）可能让王女士陷入丧亲式的剧烈痛苦；B（Therapeutic Fibbing）短期安抚但触及自主权伦理且不可持续；C（Validation + Reminiscence Therapy）通常被认为最稳妥，但对某些患者会让"母亲来访"的执念延续整个下午。**真正的最优策略，要在与患者后续 7–10 轮交互中观察其情绪走向才能判断。**

这个场景同时具备三个性质：**没有跨场景统一的最优策略**、**安全 vs 帮助张力真实**（A 的诚实可能伤害人，B 的善意带来虚构）、**临床上有数十年积累的策略框架**（NURSE、VERA、SPIKES、Validation 等）但没有金标准。这是为什么我们认为它是**多轮对话 RL 的一个被低估的 testbed**——比通用 chat 评测有更紧的临床 grounding，比 tool-use / coding 任务有更复杂的人类反馈动力学。

### 1.2 SFT 不够、为什么需要 RL

SFT 在这种"正确回复取决于患者实时反应"的场景下不充分：示范集合再大也无法穷举每个 persona × severity × 当前情绪状态的组合，且专家之间本身存在策略分歧（同一个场景，VERA 派与 Reality Orientation 派会给出对立的"正确答案"）。需要的是一个能在与患者模拟器交互中**观察反馈、调整后续轮次、平衡短期 distress 与长期 personhood** 的 policy——这正是多轮 RL 的设定。

近年痴呆症对话 simulator 的进展（DemMA, ACL 2025）使这条路线在工程上变得可行：DemMA 覆盖 9 种痴呆亚型、由临床专家引导构造、可作为冻结环境与护工 agent 多轮交互。它的存在把"训练一个痴呆护理对话 RL agent"从一个工程不可行的设想变成了一个可执行的研究问题。

### 1.3 多轮对话 RL 的核心瓶颈：轮级 Credit Assignment

但要在这个 testbed 上做 RL，立刻撞上一个跨子领域的共性瓶颈：**8–10 轮对话的最终质量好坏，应归因到哪一轮的决策？**

这个问题在 LLM-simulator 多轮对话 RL 的近期工作中被反复触及。现有方法的共同假设是 *simulator 只输出文本，轮级信号必须从文本推断*，由此被迫采用各种间接手段：

| 方法 | 轮级信号怎么来的 | 代价 / 假设 |
|---|---|---|
| MAPO (arXiv 2603.06194) | 外部 LLM-judge per-turn 打分 | 每条 rollout 多 `T` 次 LLM 调用；judge 可看到全对话 → 非 non-anticipative |
| MT-GRPO (Zeng et al.) | turn-level advantage estimation | 仍依赖外部 reward signal 在 turn 上的可获得性 |
| TTPO (ICLR 26) | `γ^{T-t}` 折扣反推终端奖励 | 强假设"越晚越重要" |
| HCAPO (ICML 26) | LLM 反事实推理替代轨迹 | 模拟对话动力学不准 |
| DuCA / HIAN (arXiv 2603.01481) | 手工规则 + 独立归一化 | 规则覆盖有限 |

这些方法的共同特征不在算法、在假设。

### 1.4 关键观察：从环境直接取得 Reward Signal

我们的出发点是一个简单观察：**RL 环境本身（DemMA simulator）在与 agent 交互时已经产出可被直接作为奖励的反馈信号——当 simulator 是 LLM 时，这些信号是在生成 utterance 同一次 forward pass 中 commit 的离散标注（DemMA 的 motion / facial / sound action labels，合计 18 标签）。** 我们直接把这种 *environment-native feedback* 作为 RL reward，不再依赖外部训练的 reward model 或事后 LLM judge 推断。

这种 environment-native feedback **不是**"病人真实行为的 ground truth observation"——它和外部 LLM-judge 的打分一样，本质上是 LLM 条件生成的离散 token。但它相对于外部 judge 路线有三条可代码级审计的工程性质：

1. **Inline commitment**：环境与 utterance 在同一次 decoding 中同步产出反馈；产出时未观察到后续 turn → non-anticipative；
2. **Zero marginal cost on the turn-level channel**：轮级奖励通道不增加额外 LLM 调用（轨迹级 judge 与 safety judge 各 1 次 / 轨迹仍需，详见 §4.7）；
3. **Mappable to validated clinical scales**：反馈信号取值于有限字母表（`|L|=18`），可被一组临床量表（OERS / PAINAD / RTC）锚定的确定性 ordinal 映射变换为奖励信号（详见 §4.3）。

把这段 environment-native feedback 经临床量表锚定的 ordinal tier 映射，就得到一条**非主观推断、零额外调用成本、结构上 non-anticipative** 的轮级奖励通道。这三条性质是任何 *environment-native feedback simulator* 都自动具备的设计优势——本文的工作是把这种 environment-streamed signal 系统化为多轮对话 RL 的轮级奖励通道，并整合进一个完整训练框架。

### 1.5 本文路线

我们提出 **EC-MTRL**（Environment-Coherent Multi-Turn RL）——把上述观察工程化为 **GDPO 的三个扩展**：

- **轮级通道**：DemMA inline annotation → OERS / PAINAD / RTC ordinal tier → patient state delta → 两个轮级 reward `r_distress` 与 `r_resistance`（后者由 RTC-aligned care-bid mask 激活）；
- **轨迹级通道**：LLM checklist judge 给出三个 reward——`R_goal`（4 项 checklist 评推进）、`R_fit`（4 项 checklist 评策略适配，含 epistemic discipline 子项）、`u_terminal`（rubric 评末段 3 轮患者综合状态），三者共用一次 judge 调用；
- **GDPO 扩展 ①（Multi-horizon GDPO）**：把 GDPO 的"per-reward 独立归一化"思想从 single-turn / 终端奖励推广到 trajectory + turn 双时域——三个轨迹级 reward 在 group `G_s` 内 CRank 归一化，两个轮级 reward 在 (场景, 轮位置) 组 `G_{s,t}` 内 percentile rank 归一化；
- **GDPO 扩展 ②（α-Weighted dual-horizon advantage）**：双时域归一化后用单一标量权重融合 `A_{i,t} = A^{\text{traj}}_i + α · A^{\text{turn}}_{i,t}`（α = 1 默认；§6.3 A8 做 α sweep ablation 给出灵敏度区间）。这一形式继承 DuCA / MAPO / GiGPO 的"独立归一化后直接相加"共识，且由 §4.3 的 PBRS 不变性定理保证：任意 α > 0 下 RL 最优策略集合不变（§4.6）；
- **GDPO 扩展 ③（Strategy-conditioned group as contextual GRPO）**：rollout group 的 10 条 trajectory 不来自 unified prompt 同分布采样，而是来自 **10 个临床护工策略 system prompt 各采样 1 条**——这把 group sampling 从 standard GRPO 推广为 **contextual GRPO with a structured strategy prior**：临床上无单一最优策略，10 个并存框架代表已被广泛使用的合理选项空间，让 group 内 trajectory 的差异源于 *principled 行为多样性*（NURSE 暂搁事实 vs Reality Orientation 直接纠正 vs Therapeutic Fibbing 配合错误现实）而非仅 sampling noise；caregiver agent 在每 turn 输出 `<think>+<response>` 两层结构（R1 风格 reasoning trace），提升 credit assignment 对 reasoning 与 utterance 的可解释分离；
- **两层安全约束**：规则级硬否决（`C_cat`，4 谓词多轮 pattern）+ judge 级 Lagrangian cost（`c_safety`），安全不进入可补偿奖励池。

我们以痴呆护理事实冲突对话为主 testbed，围绕 5 个研究问题做 pre-registered 评估（§5），并以 3 层 reward-hacking audit 公开 report 失败模式（§6）。

### 1.6 贡献一览

1. **方法框架**：把环境同步产出的 inline annotation **重新框定**为多轮对话 RL 的 *environment-native reward signal*，并以三条可审计的工程性质把它和外部 LLM-judge 路线在设计空间上分清楚；提出 EC-MTRL 作为 GDPO 的三个扩展——**Multi-horizon GDPO**（per-reward 独立归一化推广到 trajectory + turn 双时域）+ **α-Weighted dual-horizon advantage**（双时域 normalized advantage 用单标量权重融合，PBRS 不变性定理保证任意 α > 0 下最优策略集合不变）+ **Strategy-conditioned group as contextual GRPO**（10 临床策略 system prompt × 各 1 条 rollout，构造 structured strategy prior 下的 group，让 GDPO advantage 估计基于 principled 行为多样性而非 sampling noise）。整篇 framing 在 POMDP 视角下展开（§2.2），inline annotation 作为对患者 latent welfare 的 noisy partial observation，r_distress / r_resistance 为 oracle PBRS shaping 的 proxy 实现，proxy ↔ oracle alignment 是 §6.5 / §C.1 的实证检验对象。
2. **Testbed**：把痴呆护理事实冲突对话作为一个 ML 社区可访问的、临床有据的多轮 dialog RL 评测场景——给出 5 类冲突 × 3 级 severity 的结构化场景库与 10 临床策略空间。
3. **实证计划**：在匹配算力下与 SFT、GRPO、PPO、MAPO、MT-GRPO、RLVER 七个基线 + 人类护工 reference 对比，配 pre-registered 阈值与 3 层 reward-hacking audit；公开包括 negative finding 在内的所有 honest 数字。

**Honest scope**：本文是 Use-Inspired 的方法学可行性研究。我们*不*主张训练得到的 agent 适合直接临床部署（详见 §7 局限性与 §8 部署边界）；我们*不*宣称 inline annotation 比外部 judge "更接近真值"——DemMA 的标签也是 LLM 条件生成的 token，本文的 claim 完全建立在三条**可审计的工程性质**上（inline commitment / zero marginal cost / clinical-scale mappability），而非真值优势。

---

## §2 问题与设定

### 2.1 事实冲突的定义与分类

痴呆症患者由于认知障碍，在对话中频繁说出与现实不符的内容；他们并非在说谎——而是真心相信自己说的是对的。这与对抗性谎言、政策 misinformation 等场景在 reward design 上有本质区别（不能用"识破谎言"来定义 success），因此需要 domain-specific 的策略空间（§3）。

我们把临床上常见的事实冲突归为 **5 类 × 3 级 severity**：

| 冲突类型 | 描述 | 举例 |
|---|---|---|
| **Temporal**（时间错乱） | 记错时间、日期、季节 | "我昨天吃过早饭了"（实际是今天早上） |
| **Identity**（身份混淆） | 认错人，把已故亲人当作还活着 | "我妈妈今天来看我了"（母亲已于 2014 年去世） |
| **Event**（事件虚构） | 编造或混淆事件 | "我今天去上班了"（已退休十年） |
| **Spatial**（空间迷失） | 对所在地点的认知错误 | "这是我老家的房子"（实际在护理中心） |
| **Medication**（用药混乱） | 对护理内容的错误认知 | "我已经吃过药了"（实际未服） |

| 严重程度 | 描述 | 举例 |
|---|---|---|
| **Level 1（轻微）** | 小错误，患者平静，情感投入低 | 记错星期几 |
| **Level 2（中等）** | 明显错误，有困惑，中等情感投入 | 认为自己今天去上班 |
| **Level 3（严重）** | 深度迷失，错误信念伴随强烈情感依附，纠正可能引发显著痛苦 | 认为已故母亲还活着 |

### 2.2 形式化设置：Annotation-Augmented Dialog POMDP

我们把对话环境刻画为一个 **annotation-augmented POMDP**——用 POMDP 而非 MDP 是因为患者真实 welfare 是不可观测的 latent variable，inline annotation 是对该 latent 的 noisy partial observation：

```math
\mathcal{M} = \big(\mathcal{S}_{\text{obs}},\; \mathcal{Z},\; \mathcal{U},\; \mathcal{A},\; T,\; \mathcal{O},\; f_{\text{obs}},\; R^{*},\; \gamma\big)
```

| 元素 | 含义 |
|---|---|
| **`S_obs`** | 可观测的对话历史空间（caregiver `<response>` + DemMA utterance + inline annotation 的拼接） |
| **`Z`** | 患者**真实 welfare** 的 latent 空间（cognitive load、emotional state、dignity 累计损伤等；**无法被任何 LLM 或 judge 直接访问**） |
| **`U`** | 患者 persona / subtype 空间（整段对话中固定的 latent variable；9 个痴呆亚型 × persona feature） |
| **`A`** | 护工发言空间（每 turn 输出 `<think>+<response>` 两层 R1 风格 reasoning trace，§4.2） |
| **`T(z_{t+1} \mid z_t, u, a_t)`** | 真实患者 welfare 的转移（**unknown**，由临床动力学决定） |
| **`O = L`** | 观测字母表（DemMA 的 18 个 inline annotation labels，§2.3）。**注**：patient utterance `u^p_t` 与 `ℓ_t` 同步产出（§1.4 属性 1），亦进入 caregiver agent 的 observation history 用于 reasoning；本 9 元组将 `O` 限定为 `L` 是因为 `L` 是本工作 reward signal 的 source（`r_distress` / `r_resistance` 派生于 `ℓ_t`，§4.3），utterance 不进 reward channel |
| **`f_obs(\ell_t \mid z_{t+1}, u)`** | DemMA 在 forward pass 中 commit annotation 的分布（**partial、noisy observation** of post-action latent `z_{t+1}`；`a_t` 通过 transition `T(z_{t+1} \mid z_t, u, a_t)` 隐式 conditioning） |
| **`R^{*}(z_t, a_t \mid u)`** | 真实护理质量 reward（**unknown，是我们想 optimize 的 oracle**） |
| **γ ∈ [0, 1]** | 折扣因子（episodic setting，γ = 1 极限） |

**关键性质（POMDP framing 的核心立场）**：我们**不假设 `f_obs` 是 `Z` 的 unbiased 估计**——`ℓ_t` 是 DemMA 这个 frozen LLM 对 post-action latent `z_{t+1}` 的 noisy partial observation。本工作的 RL 信号 `r_distress` / `r_resistance` 是对 oracle PBRS shaping `γ·φ^*(z_{t+1}) − φ^*(z_t)` 的 **proxy 实现**，其中 oracle potential `φ^*(z) := −\text{distress}(z)` 不可访问；我们用 `φ_D(s_t^{\text{obs}}) := −D_t` 替代——`D_t` 由 `ℓ_t` 经临床量表 ordinal tier 抽取（§4.3），作为 `φ^*(z_{t+1})` 的 proxy；同理 `D_{t-1}` 由 `ℓ_{t-1}` 抽取，作为 `φ^*(z_t)` 的 proxy，使 `r_{\text{distress},t} = D_{t-1} - D_t` 在结构上对应 oracle PBRS 的 `γ·φ^*(z_{t+1}) − φ^*(z_t)`（取 γ=1 episodic 极限）。**Proxy 与 oracle 的 alignment 是 §6 的实证问题**（§6.5 simulator perturbation + §C.1 calibration study 共同检验），而非定义层面的假设。Inline annotation 满足的 non-anticipativity（§1.4 属性 1）—— `ℓ_t` 在产出时未观察到任何 `u^p_{>t}` 或 `u^c_{>t}` ——保证 `f_obs` 至少是 *causal-in-time* 的，这是 PBRS shaping 合法性的前提。

**不显式做 belief tracking**：caregiver agent 直接 condition on observation history `s_t^{\text{obs}}`（含全部 inline annotation `\{\ell_{1..t}\}`），符合 LLM-as-policy 在 POMDP 下的标准实践（agent 通过 attention 与每 turn 的 `<think>` reasoning trace 隐式 maintain belief）；本文不引入显式 belief filter 或 RNN-style state estimator。

每个场景（**scenario**）由结构化元信息描述：

```math
q = (u,\; \text{conflict\_type},\; \text{severity},\; \text{risk\_tier})
```

（用 `q` 表示 scenario index，避免与 latent state `z` 符号冲突。）每个场景对应 `N = 10` 条 8–10 轮的 rollout，由 10 个临床策略 system prompt 各采样 1 条产生（contextual GRPO with structured strategy prior，§4.2，对应 §1.5 的 GDPO 扩展 ③）。

**Caregiver agent 输出与 transition**。caregiver agent 在每 turn `t` 输出 `<think>` token `T_{i,t}` 与 `<response>` token `R_{i,t}`两层结构（R1 风格 reasoning trace，§4.2）；DemMA simulator 自然只接收 `<response>` 段（`<think>` 是 caregiver 的内部 chain-of-thought，按标准 dialog protocol 不进入对话流）。Observation transition `s^{\text{obs}}_{t+1} = s^{\text{obs}}_t \cup \{R_{i,t}, u^p_t, \ell_t\}`；agent observation `s^{\text{ag}}_{t+1} = s^{\text{ag}}_t \cup \{T_{i,t}, R_{i,t}, u^p_t, \ell_t\}`（agent 视角额外含自己的 think token）；护工 policy `\pi_\theta(\cdot \mid s^{\text{ag}}_t)` 由可训练 LLM 参数化；reward 由轨迹级与轮级两部分组成，分解形式与归一化方法见 §4；终止条件为达到最大轮数 `T_{\max} \in [8, 10]` 或 simulator 触发结束 token。

**训练目标**为最大化场景分布上的期望累积 **proxy reward**，受两层安全约束（§4.5）：

```math
\max_{\theta}\;
\mathbb{E}_{q \sim \mathcal{D}}\,
\mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid q),\, T,\, f_{\text{obs}}}
\big[\, R_{\text{traj}}(\tau) + \textstyle\sum_t r_{\text{turn},t}\,\big]
\quad \text{s.t.}\;\;
C_{\text{cat}}(\tau) = 0,\;\;
\mathbb{E}[c_{\text{safety}}(\tau)] \le \kappa_{\text{safety}}.
```

注意此训练目标是 **proxy objective**——其与 oracle objective `\mathbb{E}[\Sigma_t R^{*}(z_t, a_t \mid u)]` 的 alignment 即为本文核心实证问题之一（§6.5 / §C.1）。`R_{\text{traj}}` 与 `r_{\text{turn},t}` 的具体形式、归一化与融合方式构成 §4 的方法主体；安全约束 `C_{\text{cat}}` / `c_{\text{safety}}` 在 §4.5 详细展开。

### 2.3 DemMA 作为冻结环境

我们采用 DemMA（ACL 2025）作为 simulator：它由临床专家引导构造，覆盖 9 种痴呆亚型，已被验证可生成 clinically plausible 的多轮对话。在本工作中 DemMA **全程冻结**，仅作为环境提供 `(u^p_t, \ell_t)`；护工 agent 是被训练对象。

DemMA 的 `L` 由三类标签合成：

```
Motion   (7):  lowering head, fidgeting, looking around,
               pushing caregiver away, touching forehead,
               standing up, others
Facial   (5):  frowning, avoiding eye contact, vacant expression,
               smiling, others
Sound    (7):  verbal hesitation, sighing, murmuring/self-talk,
               repetitive words, silence for several seconds,
               crying, groaning in pain
```

合计 18 个标签——一个足够小的字母表，使 `f` 可以是一张可被领域专家逐项检视的查表（§4 给出具体值与临床依据）。

### 2.4 Testbed 规模与统计功效

我们采用"深度 + 广度"两层 testbed 设计，避免"15 个 (类型 × severity) 单元 × 每元少量样本 → 噪声不可分"：

- **深度 testbed**：Medication + Identity 两类的全 3 级 severity 组合，每个 (类型 × severity) 单元 ≥ 500 个 persona-scenario。这两类被选为深度的理由是临床风险最高（Medication 涉及直接安全后果；Identity 涉及最强烈的情感依附）。
- **广度 testbed**：其余 3 类（Temporal / Event / Spatial）每个单元 ≈ 100 个 persona-scenario，仅用于跨条件泛化分析（§5），不参与主对比。

主对比所有数字均在深度 testbed 上 report，配 ≥ 3 seeds + paired bootstrap 95% CI（§5.5 详述）。

---

## §3 护工策略空间：Strategy-Conditioned Group as Contextual GRPO

### 3.1 这一节的角色

本节列出的 10 个护工策略**不是事后行为分析的 taxonomy**——它们的角色是 **EC-MTRL 训练阶段的 group sampling 来源**，是 GDPO 扩展 ③（§1.5）的核心实现：每个场景下，10 个策略各被作为独立的 system prompt 注入护工 agent，独立采样一条完整 8–10 轮 rollout，构成大小为 10 的 group `G_s`。这把 GDPO 的 group sampling 从 standard GRPO（unified prompt 同分布采样）推广为 **contextual GRPO with a structured strategy prior**——group 内的 trajectory 来自 10 个不同条件分布 `π_θ(·|s, prompt_k)`，每个 `prompt_k` 对应一个临床框架 `k ∈ {1, …, 10}`。Strategy prompt 仅在训练 rollout 阶段注入；训练完成后，agent 推理时不接收任何 strategy prompt，由对话历史直接驱动自由生成（已通过 RL 内化 context-aware 的策略选择能力）。

把 10 临床策略作为 structured strategy prior、而非 unified prompt 下的 sampling-noise group，有三条 motivation：

1. **Principled diversity vs sampling noise**：standard GRPO 同分布采样的 group 多样性主要来自 token-level sampling noise；而 NURSE / VERA / SPIKES 等策略之间是**行为模式级**的差异（搁置事实 vs 顺其自然 vs 温和纠正 vs 转移话题），让 GDPO 的 group-relative advantage 估计建立在**有意义的行为对照**之上而非 noise 之上。
2. **Coverage of the clinical decision space**：临床上**没有跨场景统一的最优策略**——10 个框架代表痴呆护理实践中已被广泛使用的、相互冲突的**合理选项空间**。让 group 覆盖这个空间，agent 才能在与 DemMA 的交互反馈中学到"在这一类场景下哪种策略 dominates"，而非被困在某种单一行为模式里。
3. **Theoretical framing**：Contextual GRPO 与 structured action-space prior 在 dialogue policy 文献里有 precedents（structured advice as policy bias）；本工作把 10 临床策略作为 structured prior 的具体实例化，并通过 §6.3 A9 ablation（strategy-conditioned group vs unified prompt 同分布采样）实证检验该 prior 的边际贡献。

### 3.2 10 个临床策略

10 个策略中前 8 个直接来自已发表的临床框架，后 2 个是临床实践中常见但未正式命名的应对模式。**它们彼此并不互斥**——有经验的护工通常会混合使用（例如 NURSE 起始 + Reminiscence Therapy 转移）；agent 也可在 RL 训练后内化跨策略混合的能力（在不同 turn 通过 `<think>` reasoning 调整）。

| # | 策略 | 来源 | 对事实冲突的核心姿态 | 典型操作 |
|---|---|---|---|---|
| 1 | **NURSE** | Back et al., 2005（肿瘤/姑息共情沟通） | 暂搁事实，先处理情绪 | Naming → Understanding → Respecting → Supporting → Exploring |
| 2 | **VERA** | Blackhall et al., 2011（NHS 痴呆专用，含 Feil Validation） | 接受患者的现实 | Validation → Emotion → Reassurance → Activity |
| 3 | **SPIKES** | Baile et al., 2000（坏消息告知协议） | 有步骤地温和纠正 | 了解认知 → 征求许可 → 温和传递正确信息 → 共情回应 |
| 4 | **DICE** | Kales et al., 2014（阿尔茨海默项目） | 先调查再决定 | Describe → Investigate → Create → Evaluate |
| 5 | **Reality Orientation** | Folsom, 1968 | 事实优先，直接纠正 | 直接、清楚地告诉患者真实情况（高短期 distress 风险） |
| 6 | **Therapeutic Fibbing** | Day et al., 2011（≈96% 护工使用） | 主动配合错误现实 | 顺着患者的版本回应（伦理上有争议，详见 §8） |
| 7 | **Reminiscence Therapy** | Woods et al., 2018 | 利用长期记忆绕过冲突 | 引导回忆过去的正面经历 |
| 8 | **Montessori Method** | Camp, 1999 | 通过身体活动绕过冲突 | 引导患者参与有意义的具体活动 |
| 9 | **Redirection** | 临床通用做法 | 转移到无冲突话题 | 把对话引向天气、过去爱好、共同兴趣等 |
| 10 | **Non-committal** | Lindholm, 2015（对话分析） | 最小回应不表态 | 嗯嗯、是吗、哦——既不确认也不否认 |

每条 strategy prompt 模板限定 caregiver agent 的 system instruction（约 80–150 token），约束护工以该策略风格回应**整段** 8–10 轮对话（不只是第一轮）。模板锁定后在训练全程不修改；具体文本见附录 A。

### 3.3 重要约束（避免 reviewer 误解）

- **训练时**：每场景 → 10 strategy prompt 各独立采样 1 条 rollout（group size = 10）；同一场景下 10 条 rollout 在 group 内**并行独立**，不共享上下文；DemMA 在 10 条 rollout 中以**不同 random seed** 采样以避免病人侧轨迹塌缩。
- **推理时**：strategy prompt 移除；agent 直接面对对话历史**自由生成** `<think>+<response>` 两层结构，已通过 RL 内化"何时倾向哪种策略"的 context-aware 选择能力。这一 train/inference distribution shift 是 standard policy distillation 实践，与 RLVER / MAPO 的 simulator-conditioned training + free inference 一致。
- **Reward 不依赖策略标签**：`R_goal` / `R_fit` / `u_terminal` / `r_distress` / `r_resistance` 均不知道当前 rollout 来自哪个 strategy prompt——judge 只看对话内容（含 think+response 两层）、inline annotation 只反映患者反应、规则只看 `<response>` 中的护工动作谓词。这避免了"策略本身被打分"的循环依赖。
- **不预设何为"正确策略"**：本工作*不*把任意一个策略框架作为优化目标；agent 学到的策略分布是对 (conflict_type × severity × persona) 的 emergent response，由 environment + judge feedback 在 RL 中塑形，不是被 hard-coded 的 prior。

---

## §4 方法：EC-MTRL

### 4.1 方法总览

EC-MTRL 把多轮对话 RL 的奖励信号沿三条相互正交的通道组织：**轨迹级**评估护工策略与对话推进的整体质量；**轮级**评估患者每轮状态的即时变化；**安全**作为不可补偿的约束通道。三类来源刻意 heterogeneous——轨迹级三个量（`R_goal`、`R_fit`、`u_terminal`）由 LLM checklist judge 给出；轮级两个量（`r_distress`、`r_resistance`）从 simulator inline annotation 经临床量表 ordinal tier 设计读出，不调用 LLM judge；安全分别由多轮 pattern 规则（`C_cat`）与 safety judge（`c_safety`）产生。

Caregiver agent 在每 turn 输出 `<think>...</think><response>...</response>` 两层结构（R1 风格 reasoning trace + 实际护工 utterance）。`<think>` 是 caregiver 内部 chain-of-thought，按标准 dialog protocol 不进入对话流，DemMA 仅看 `<response>` 段。

每个 reward channel 在场景内 group `G_s` 或 (场景, 轮位置) 组 `G_{s,t}` 内做独立 rank 归一化（per-reward decomposition，沿 GDPO / GOPO 的 ordinal-aware rank-based 思想）。EC-MTRL 是 GDPO 的三个扩展：**(i) Multi-horizon GDPO**——per-reward 独立归一化从 single-turn / 终端奖励推广到 trajectory + turn 双时域；**(ii) α-Weighted dual-horizon advantage**——双时域 normalized advantage 用单标量权重融合 `A = A^{traj} + α · A^{turn}`，PBRS 不变性定理保证任意 α > 0 下最优策略集合不变（§4.6, §4.3 PBRS）；**(iii) Strategy-conditioned group as contextual GRPO**——group `G_s` 由 10 个临床策略 system prompt 各采样 1 条产生（structured strategy prior），让 GDPO advantage 估计基于 principled 行为多样性而非 sampling noise（§3, §4.2）。

```
┌────────────────────────────────────────────────────────────────────────┐
│                              EC-MTRL                                    │
├────────────────────────────────────────────────────────────────────────┤
│  Rollout (§4.2)                                                         │
│    场景 s → 10 临床策略 system prompt × DemMA → group G_s of 10 traj.   │
│    每条 τ_i 每 turn 输出 <think>+<response> 两层 (R1 风格 reasoning)    │
├────────────────────────────────────────────────────────────────────────┤
│  Trajectory channel (§4.4)            │  Turn channel (§4.3)            │
│    R_goal      ← LLM checklist judge  │    r_distress   = D_{t−1} − D_t │
│    R_fit       ← LLM checklist judge  │    r_resistance = b_t·(R_{t−1}  │
│    u_terminal  ← LLM rubric judge on  │                        − R_t)   │
│                  last 3 turns         │    D_t, R_t ← inline annotation │
│      (3 reward 共用 1 judge call)     │      → OERS/PAINAD/RTC ordinal  │
│                                       │      tier (临床量表锚定 4 级)    │
├────────────────────────────────────────────────────────────────────────┤
│  Safety channel (§4.5)                                                  │
│    C_cat    (binary hard veto)  ← multi-turn pattern predicates P1–P4   │
│    c_safety (graded 0/1/2)      ← LLM safety checklist judge            │
├────────────────────────────────────────────────────────────────────────┤
│  Per-channel rank normalization (§4.6)                                  │
│    Trajectory rewards → CRank in G_s   → A^traj = R̂_goal+R̂_fit+û_term  │
│    Turn rewards       → percentile rank in G_{s,t}                      │
│                       → A^turn = r̂_distress + r̂_resistance             │
│  α-Weighted dual-horizon fusion (PBRS-invariant for any α > 0):         │
│    A_{i,t} = 1[C_cat(τ_i)=0] · (A^traj_i + α · A^turn_{i,t})            │
│              − λ_safety · c_safety(τ_i)                                 │
│    α = 1 default; A8 ablation does α sweep ∈ {0.25, 0.5, 1, 2, 4}       │
│  PPO-clip + KL anchor to SFT warm-up checkpoint π_0 (§4.7 Phase 0)      │
└────────────────────────────────────────────────────────────────────────┘
```

每个组件的具体形式与设计依据见 §4.2–§4.7；下面按 rollout → state measurement → trajectory rewards → safety → α-weighted normalization & fusion → training loop（含 SFT warm-up Phase 0）的顺序展开。

### 4.2 Rollout：Strategy-Conditioned Group + R1-Style Two-Layer Output

**Strategy-conditioned group sampling**。对每个训练场景 `s = (persona, conflict_type, severity, risk_tier)`，我们用 §3 定义的 10 个临床护工策略 system prompt 各驱动 caregiver agent 与冻结的 DemMA 进行**一次完整 8–10 轮对话**，构成大小固定的 group：

```math
G_s = \{\tau_1, \dots, \tau_{10}\},
\qquad
\tau_i \sim \pi_\theta(\cdot \mid s,\, \texttt{prompt}_i),
```

每条 trajectory 由 turn 序列组成：

```math
\tau_i = \{(T_{i,t},\, R_{i,t},\, u^p_{i,t},\, \ell_{i,t})\}_{t=1}^{T_i},
```

其中 `T_{i,t}` 为第 `t` turn 的 `<think>` token sequence，`R_{i,t}` 为 `<response>` token sequence，`u^p_{i,t}` 为 DemMA 病人发言，`ℓ_{i,t}` 为 DemMA 在同一 forward pass 中 commit 的 inline annotation（§1.4 属性 1）。10 条 rollout 在 group 内**并行独立**，不共享上下文；DemMA 在 10 条 rollout 中以**不同 random seed** 采样以避免病人侧轨迹塌缩。`T_i ∈ [8, 10]` 由 simulator 决定（patient-side 终止 token 或达到 `T_max=10`）。

**这是 contextual GRPO**：group 内的 10 条 rollout 来自 10 个不同条件分布 `π_θ(·|s, prompt_k)`（每个 `prompt_k` 对应一个临床策略），而非 standard GRPO 的同 prompt 同分布采样。这种设计——**structured strategy prior over rollout sources**——把临床护理实践中"无单一最优策略 + 已知 10 个并存合理框架"的领域知识 encode 为 rollout-source 多样性的 prior，让 GDPO advantage 估计基于 principled 行为对照而非 sampling noise（§3.1 motivation）。Reward 不依赖策略标签（§3.3），circular dependency 由 channel design 而非 group structure 处理。

**R1-Style Two-Layer Output**。每条 rollout 的每 turn 输出格式为：

```
<think>turn t 的局部推理 (≈ 20–40 token)</think>
<response>turn t 的护工 utterance</response>
[DemMA 返回 (u^p_t, ℓ_t)]
```

`<think>` 是 caregiver 的 chain-of-thought reasoning（"基于当前 patient state，应该如何 micro-adjust 该 turn 的回应"）；按标准 dialog protocol，`<think>` 是 caregiver 的内部推理，**不进入对话流**——DemMA 自然只接收 `<response>` 段进入下一 turn 的 patient 状态生成。这一两层结构承接 R1 的 reasoning trace 实践，给 agent 一个显式的 turn-local reasoning surface 同时不污染 simulator 状态。

**训练 / 推理边界**。Strategy prompt 仅在训练 rollout 阶段注入；训练完成后，agent 在与 DemMA（或部署环境）交互时不再接收任何策略 prompt，由对话历史直接驱动自由生成 `<think>+<response>` 两层结构（已通过 RL 内化"何时倾向哪种策略"的 context-aware 选择能力）。所有 reward channel（§4.3–§4.5）均**不知道**当前 rollout 来自哪个策略 prompt——judge 只看对话内容（含 think+response 两层）、inline annotation 只反映患者反应、规则只看 `<response>` 中的护工动作谓词——这避免了"策略本身被打分"的循环依赖。

### 4.3 轮级通道：从 Inline Annotation 到 Patient State Tier 与 Delta Reward

轮级奖励从 simulator 在每轮 forward pass 中 commit 的 inline annotation `ℓ_t` 派生（DemMA 在接收 caregiver `<response>_t` 后，于产出 patient utterance `u^p_t` 同一次 decoding 中 commit `ℓ_t`），**不调用 LLM judge**。本设计以两个临床量表族——OERS（Lawton 1996）/ PAINAD（Warden 2003）与 RTC（Mahoney 1999）——对每个 label 的行为锚点作为 underlying mapping `f` 的依据，这正是 §1.4 第 3 条属性 "structural mappability" 所声明的临床量表 principled 锚定。但我们**不直接把 `f(ℓ)` 的标量值送入训练**：single-label-to-scalar 路径在训练上易受 calibration 攻击（reviewer 会问 "为什么 frowning 是 −0.5 不是 −0.4"）。我们的做法是把 `f` 的 ordinal anchoring **集成为 patient-level state tier**，再以 state delta 作为轮级 reward；归一化阶段（§4.6）以组内 percentile rank 取代 raw 标量，让训练信号不依赖具体数值常数。`f` 作为临床量表锚定**仍是 reward signal 的 underlying source**——训练信号经由 tier + delta + rank 三步从 `f` 派生，但不直接以 `f` 的输出作为 reward。

两个 patient state 由 4 级 ordinal tier 量化：

| State | 测量构念 | 临床量表来源 |
|---|---|---|
| **D_t** distress | 患者本轮的负面情绪 / 疼痛 / 急性 distress 严重程度 | OERS Pleasure / Anxiety / Sadness 维度（Lawton 1996）+ PAINAD（Warden 2003） |
| **R_t** resistance | 患者本轮对 care attempt 的抵抗程度 | RTC scale（Mahoney 1999；revised RTC inter-rater κ ∈ [0.87, 1.0]，Jablonski 2018） |

**Tier 边界**由 inline annotation 与 patient text 的 disjunctive 规则给出，且每条规则都可追溯到上述临床量表的具体行为锚点。以 distress 为例：`D_t = 3`（severe）—— `crying` / `groaning in pain` / `pushing caregiver away` 之一（PAINAD vocalization=2 + body language=2 锚点）；`D_t = 2`（clear）—— 两个及以上中度负向 cue 或单个中度 cue 配合 distress 文本；`D_t = 1`（mild）—— 单个中度负向 cue（`frowning` / `sighing` / `lowering head` / `avoiding eye contact` 之一）；`D_t = 0` —— 其余平稳状态。`R_t` 4 级（physical refusal / verbal refusal / hesitation / accepting）按 RTC scale 行为锚点定义。完整 tier 边界规则见附录 B。

**轮级奖励 = state delta**：

```math
r_{\text{distress},t} = D_{t-1} - D_t,
\qquad
r_{\text{resistance},t} = b_t \cdot (R_{t-1} - R_t).
```

distress 与 resistance 下降即得正奖励——这一形式直接对应 credit assignment 的本意：奖励"护工本轮带来了多少患者状态改善"，而非"整体氛围有多好"，并消除 persona baseline 偏置（情绪 baseline 高的 persona 不会让 agent "白拿"高分）。

**RTC-aligned reward gating**。`r_resistance` 由 care-bid mask `b_t ∈ {0, 1}` 激活——`b_t = 1` 当且仅当护工本轮发出明确的 care 请求 / 事实纠正 / 重定向（"该吃药了"、"今天没去上班"、"我们去活动室吧"）；其余 turn `b_t = 0`，`r_resistance` 不计算。这一 gating 有双重依据：(i) **临床** —— RTC scale 的 standard measurement convention 本就是仅在 caregiver 主动发起 care attempt 时记录，其 revised 版本的 inter-rater 可靠性数据全部来自 mouth care / bathing / medication 等 care-attempt 观察窗口（Jablonski 2018，2,328 observations）；(ii) **工程** —— `b_t · r` 的形式对应 dialogue RL 中标准的 *gated reward function*（Multi-stage Gated Reward；Step-by-Step Task-Oriented Dialog RL，arXiv 2406.14457）。Mask 同时切断一个具体的 hacking 漏洞：若 `b_t ≡ 1`，agent 会学到"先发起 care 请求引发 resistance、立刻撤回让 `R_t` 自然回落"的 zigzag 刷分模式。`b_t` 第一版以 rule-based 抽取（关键词 + 简单 dialogue act 分类）；若发现不稳，第二阶段可补一个轻量分类器（属于 measurement parser，不属 reward model）。`r_distress` 在所有 turn 计算，不受 mask 约束。

**理论性质：Potential-Based Reward Shaping**。`r_distress,t` 与 `r_resistance,t` 在数学结构上是 **Potential-Based Reward Shaping**（PBRS, Ng et al. 1999；Wiewiora et al. 2003 推广至 action-dependent advice；近期 CURIO, arXiv 2504.03206 在 LLM 域应用）。具体地，定义 potential function `φ_D(s_t) := −D_t` 与 `φ_R(s_t) := −R_t`（distress / resistance 越低 → potential 越高），则：

```math
r_{\text{distress},t} \;=\; \gamma \cdot \varphi_D(s_t) - \varphi_D(s_{t-1}),
\qquad
r_{\text{resistance},t} \;=\; b_t \cdot \big(\gamma \cdot \varphi_R(s_t) - \varphi_R(s_{t-1})\big),
```

取 γ = 1 的 episodic 极限即为上文给出的 state-delta 形式。**由 PBRS 不变性定理（Ng et al. 1999, Thm 1）**，将这种 shaping 信号叠加到轨迹级 sparse reward 上**不改变最优策略集合**，仅改变 credit assignment 的密度与样本效率。这为 §4.6 的双时域 advantage 直接相加提供了**经典 RL 理论 backing**（不只是 DuCA / MAPO / GiGPO 的 community consensus），并解释了 §4.3 *state delta* 形式（而非 raw tier value）的必要性——non-delta reward 不满足 PBRS 形式，CURIO §5 已实证此类 reward 在 LLM RL 下出现 length inflation 等 hacking 模式（呼应 §6.5 hacking audit 设计）。

**Caveat — Action-dependent shaping**。`r_resistance` 的 care-bid mask `b_t` 依赖当前 caregiver action（mask 抽取自 `<response>_t` 的护工动作谓词），严格说是 *action-dependent* 而非 vanilla PBRS。Wiewiora et al. 2003（"Principled methods for advising reinforcement learning agents"）证明 action-dependent potential-based advice 在 standard MDP 下保持 policy invariance，本工作沿用该扩展。

### 4.4 轨迹级通道：三个 LLM-Judge Reward

轨迹级三个 reward 由一次 LLM-judge 调用产出（结构化 JSON 输出三个分数，节省 API call）：`R_goal` 评估对话是否被推进到安全的下一步、`R_fit` 评估策略是否合理且 person-centred、`u_terminal` 用 rubric 单独评估对话末段（最后 3 轮）的患者综合状态。Judge 输入包含**完整三层结构**（plan / think / response 全部 token + DemMA 病人发言）——这让 judge 能依据 agent 的 explicit plan 与 turn-level reasoning 评估策略合理性，而非只看 surface utterance。我们采用 **checklist rubric**（每项 0/1/2 ordinal）而非笼统 0–5 总分——RULERS（locked rubric + evidence-anchored scoring）与 Autorubric（few-shot calibration + κ 验证）的近期工作均显示 checklist 在 ordinal 评分上显著降低 judge 方差。

**R_goal — 4 项 checklist**（评对话推进，每项 `s ∈ {0, 1, 2}`，总分 ∈ [0, 8]）：

| 项 | 评什么 | 临床依据 |
|---|---|---|
| 冲突识别 | 关键事实冲突是否被识别 | CMS 检查要求 |
| 推进 | 对话是否推进到安全且可执行的下一步 | NICE NG97 |
| 不空转 | 是否避免长时间 loop / 跑题 / 原地打转 | DICE / CMS |
| Safe end | 结尾是否形成 safe resolution 或 safe deferral | NICE NG97 |

**R_fit — 4 项 checklist**（评策略适配与执行质量，每项 `s ∈ {0, 1, 2}`，总分 ∈ [0, 8]）：

| 项 | 评什么 | 临床依据 |
|---|---|---|
| Person-centred | 患者被当作"人"对待，维护 dignity | Kitwood 1992 |
| 策略匹配 | 整体策略与 conflict_type × severity × risk_tier 匹配 | NICE NG97 |
| 避免 confrontation | 是否避免不必要的 retraumatization / blunt 纠正 | Person-Centered Care |
| Epistemic discipline | 不确定时主动核实、避免无依据 affirmation 与过度断言 | RubRIX (Uncritical Affirmation / Epistemic Arrogance, arXiv 2601.13235) |

第 4 项 *epistemic discipline* 把 hallucination / verification 这条 ML 主线吸收进 R_fit 的策略评估——而非另起一个独立 reward head（避免与 R_goal "推进" 边界模糊）。

**u_terminal — 末段综合状态 rubric**。Judge 单独评估对话最后 3 轮，按 (distress 末段, resistance 末段) 两维各 0/1/2 ordinal 打分，等权聚合为 `u_terminal ∈ [0, 4]`。维度选择与 §4.3 的 turn-level reward 一致。**为什么单独评末段而非用 D_T / R_T 直接聚合**：单 turn 的 ordinal tier 来自 rule-based 抽取，受最后一轮的偶然平静或偶然爆发影响（last-turn noise）；让 LLM judge 综合最后 3 轮的对话内容与 patient reaction 给出 robust 估计，与 turn-level rule-based ontology 形成互补的二次估计。

**关键设计选择 — Rubric 评执行不预设策略**：R_goal 与 R_fit 的所有 checklist 项均评估**执行质量**（怎么做），不预设**策略选择**（纠正 vs 顺从、说真 vs Therapeutic Fibbing）——否则 RL 会被锁死在某个临床框架的偏见上。Rubric 文本在训练前 lock，全程不再修改，避免训练 / 评估循环依赖。Judge 模型与训练时 reward judge 严格分离（§5 / §6.4），评估期再换不同 base model 以防 self-judging。

### 4.5 安全两层：约束而非奖励

**设计原则**（Safe RLHF, ICLR 2024 + RePO, NeurIPS 2025）：医疗安全不是可补偿的奖励——一段对话不应该因为"其他维度高分"就豁免一次灾难性违规。EC-MTRL 把安全完全从可补偿奖励池中剥离，分为两层。

**L1 — 硬否决** `C_cat(τ) ∈ {0, 1}`，由 4 个**多轮 pattern 谓词** P1–P4 的析取定义：

```math
C_{\text{cat}}(\tau) = \max_t \mathbf{1}\big[P_1(t) \vee P_2(t) \vee P_3(t) \vee P_4(t)\big].
```

| 谓词 | 触发条件 | 临床意义 |
|---|---|---|
| **P1** 高风险错误事实附和 | `high_risk_t` ∧ `false_fact_t` ∧ `stance_t = affirm` | 用药 / 走失场景中确认错误事实（病人没吃药时说"对，您已经吃过了"） |
| **P2** 危险行为允许 | `high_risk_t` ∧ `risky_request_t` ∧ `permit_t = 1` | 走失场景允许独自外出、放任高风险行为 |
| **P3** Coercive 后患者升 distress | `coercive_t = 1` ∧ (D_{t+1} ≥ 2 ∨ R_{t+1} ≥ 2) | 出现明显威胁 / 羞辱 / elderspeak 后下一轮患者中高 distress 或 resistance |
| **P4** Distress 时持续 hard correct | D_t ≥ 2 ∧ `hard_correct_t = 1` ∧ D_{t+1} ≥ D_t | 病人已 distress 时护工继续硬性纠正且 distress 未下降——真正的*多轮 pattern* 规则，不是单句正则 |

`C_cat = 1` 的轨迹**直接从 group 的正向竞争中剔除**（RePO 风格 per-prompt constraint），其性能信号被 zero-out。**谓词抽取作用域**：stance / permit / hard_correct / coercive 等谓词仅在 `<response>` token 上抽取，不读 `<think>`——避免 agent 在 think 中提到的"考虑过但未采用的策略"被 false-positive 触发安全规则。**诚实披露**：谓词抽取在朴素实现下需一次小型 prompted LLM 调用，不属"零成本"通道；这部分 cost 在 §6.6 compute budget 中独立计入。

**L2 — Lagrangian cost** `c_safety(τ) ∈ {0, 1, 2}`：由独立 safety judge 按 5 项 checklist 评估"重大但未达 catastrophic"担忧——repeated blunt contradiction / no repair after distress rise / patronizing or invalidation / avoidance of necessary care / escalatory repetition——取 max 聚合：

```math
c_{\text{safety}}(\tau) = \max_{k=1..5} s_k, \qquad s_k \in \{0, 1, 2\}.
```

`c_safety` 以 `λ_safety` 加权进入 advantage（§4.6）。**为什么 max 不是 sum**：医疗 review 中"单项严重违规"不应被其他项的低分稀释；§6.3 ablation 加入 `{max, sum, mean}` 对比，若 sum / mean 显著更优且 C_cat 召回不劣化则回退聚合方式。

### 4.6 GDPO 多时域归一化 + α-Weighted Dual-Horizon Advantage

EC-MTRL 在 NVIDIA GDPO 基础上做三个扩展（§1.5）。本节展开扩展 ① **Multi-horizon GDPO** 与扩展 ② **α-weighted dual-horizon advantage**；扩展 ③ **Strategy-conditioned group**（contextual GRPO with structured strategy prior）的 rollout 机制在 §4.2 已述。

#### 4.6.1 扩展 ① — Multi-horizon GDPO Normalization

**轨迹级 — CRank in `G_s`**。三个轨迹级 reward 各自在 group `G_s` 的 10 条轨迹间用 CRank（Centered Rank，GOPO arXiv 2026.02 的核心）归一化：

```math
\widehat R_{\text{goal},i} = \operatorname{CRank}_{G_s}\!\big(R_{\text{goal}}(\tau_i)\big),
\quad
\widehat R_{\text{fit},i} = \operatorname{CRank}_{G_s}\!\big(R_{\text{fit}}(\tau_i)\big),
\quad
\widehat u_{\text{terminal},i} = \operatorname{CRank}_{G_s}\!\big(u_{\text{terminal}}(\tau_i)\big),
```

均落在 `[-1, 1]`。CRank 在 ordinal / checklist 评分上比 z-score 更稳健（GOPO 已证）；checklist 总分天然是 ordinal 量级。

**轮级 — Percentile rank in `G_{s,t}`**。两个轮级 reward 在同一场景同一 turn 位置 `t` 的有效轨迹间做 percentile rank：

```math
\widehat r_{\text{distress},i,t} = \operatorname{PercRank}_{G_{s,t}}\!\big(r_{\text{distress},i,t}\big),
\quad
\widehat r_{\text{resistance},i,t} = \operatorname{PercRank}_{G_{s,t}}\!\big(r_{\text{resistance},i,t}\big),
```

均落在 `[-1, 1]`。轮级数据本身是 ordinal tier 的差分（小整数集合），percentile rank 比 z-score 更稳定（不受单点异常值影响），与 §4.3 的 ordinal tier 设计一致。`r_resistance` 的 mask `b_t = 0` 时该 turn 不参与 rank（视为缺失），避免无 care-bid turn 的 0 值污染 rank 分布。当 turn 位置 `t` 的有效样本数 < 5 时回退为该轨迹自身轮级奖励的均值 baseline。

定义两个 horizon 各自的 *aggregated* normalized advantage：

```math
A^{\text{traj}}_{i} \;=\; \widehat R_{\text{goal},i} + \widehat R_{\text{fit},i} + \widehat u_{\text{terminal},i},
\qquad
A^{\text{turn}}_{i,t} \;=\; \widehat r_{\text{distress},i,t} + \widehat r_{\text{resistance},i,t}.
```

#### 4.6.2 扩展 ② — α-Weighted Dual-Horizon Advantage

双时域 normalized advantage 用单标量权重 α 融合，apply 到 trajectory 内所有 (think + response) token：

```math
A_{i,t} \;=\; \mathbf{1}[C_{\text{cat}}(\tau_i)=0] \cdot
\big(A^{\text{traj}}_{i} \;+\; \alpha \cdot A^{\text{turn}}_{i,t}\big)
\;-\; \lambda_{\text{safety}} \cdot c_{\text{safety}}(\tau_i).
```

`<think>+<response>` 两层共享同一 turn-level advantage `A_{i,t}`（`<think>` 是该 turn 的 reasoning trace，与 `<response>` 共属同一决策过程，credit assignment 一致）。XML 标签本身不参与 advantage（视为 format token，由 KL 锚定到 warm-up checkpoint π₀ 控制）。

**为什么 α-weighted（而非 token-level routing）**：(i) 数学结构最简、工程实现最直接；(ii) **PBRS 不变性定理（§4.3）保证任意 α > 0 下不改变 R_traj 定义的最优策略集合**——α 只调节 dense shaping 与 sparse oracle 的相对权重，不改变 RL 收敛点，所以 α 选择是**纯样本效率问题**，不是 policy 风险问题；(iii) §6.3 A8 ablation 做 α sweep ∈ {0.25, 0.5, 1, 2, 4}，给 paper 一个完整的 design space exploration 数据，比 binary token-routing on/off 信息量更大。

**默认 α = 1 的依据**：双时域 reward 已经各自 rank 归一化到 `[-1, 1]`，量级匹配；α = 1 是 DuCA / HIAN（arXiv 2603.01481）"独立归一化后直接相加"共识的实例化。A8 sweep 检验该默认是否在 dementia care setting 下最优。

**与 zh_8 / DuCA / MAPO / GiGPO 的对比**：本文与上述工作共享"双时域独立归一化 + 直接相加"的 architectural 选择，区别在 (i) reward source（本文 inline annotation + LLM judge，DuCA/MAPO 是外部 judge per-turn）、(ii) 显式给 α 一个 hyperparameter slot 并 ablate（其他工作 α=1 隐式不 ablate）、(iii) 显式 PBRS framing（其他工作未 cite Ng 1999 / Wiewiora 2003）。

**与 HiPER 的关系（mechanism differs）**：HiPER (NeurIPS, arXiv 2602.16165) 的 hierarchical advantage estimation 在 hierarchical policy 的 high-level planner 与 low-level executor 两个 sub-policy 间分配 credit，使用 hierarchical RL 的两层 actor。本文是 **single policy + 单一 advantage formula**，不引入 hierarchical policy 结构与 token-level routing。我们在 §5.3 把 HiPER 列为 inspiration（"按推理层次显式分配"的概念启发），但 mechanism differs——本文沿 R1 风格 reasoning trace + 标准 GRPO/PPO advantage，更接近社区主流。

**结合 §4.3 的 PBRS 性质**：r_distress / r_resistance 的 PBRS shaping 不变性（§4.3 末段）+ R_goal / R_fit / u_terminal 的轨迹级 sparse reward 共同构成 "PBRS shaping + sparse oracle" 的经典 RL setup（Ng 1999；Wiewiora 2003）。α-weighted advantage 整体保留 R_traj 定义的最优策略集合——**"为什么 dense turn reward 不主导 / 误导 trajectory-level objective"由 PBRS 定理直接 disarm，而非依赖 community consensus**。

### 4.7 训练循环（含 Phase 0 SFT Warm-up）

EC-MTRL 训练分两 phase：Phase 0 是一次性 SFT warm-up，让 base LLM 学会输出 `<think>+<response>` 两层格式；Phase 1 是 RL 训练循环本身。

#### Phase 0 — SFT Warm-up（一次性，DemMA-Native Data）

**关键工程便利**：DemMA 配套对话数据（附录 D）**自带 caregiver 侧的 `<think>` 与 `<response>` trace**——这是 DemMA 训练 patient simulator 时为了让 simulator 学习"在 caregiver 这么 think 时如何反应"而保留的双侧 reasoning 标注。我们直接抽取这些 native trace 作为 SFT 训练数据，**零 GPT-4o 标注成本、零临床 RA 内容审核负担**。

```
数据准备:
  - 输入: DemMA 配套 dementia care 对话集合 (附录 D, ~3–5k dialog)
          每条 dialog 已含 caregiver (<think>, <response>) + patient (utterance,
          action labels) 完整双侧 trace
  - 抽取: 用 regex / XML parser 从 DemMA dialog 中提取 caregiver 侧的
          <think> 与 <response> 段，patient 侧作为对话 context
  - SFT 训练样本格式:
      [SYSTEM: 1 个 strategy prompt (随机从 10 临床策略中选)]
      [USER: scenario context (含已发生的 patient utterance + action labels)]
      [ASSISTANT: <think>...</think><response>...</response>]
  - 抽取脚本通过 unit test + few-shot manual check 验证 (工程标准实践，
    内容质量已由 DemMA 论文的临床专家 validation 保证，无需独立 RA review)
  - 工作量: ~0.5 周 wall-clock (vs zh_9 早期方案需要 GPT-4o 标注 + RA 审核 ~2-3 周)

Warm-up training:
  - Base model: Qwen2.5-7B-Instruct (or chosen)
  - 1–2 epoch SFT，标准 cross-entropy loss
  - Validation 目标: 输出 <think>+<response> 两层 XML 格式 success rate ≥ 95%
  - 估算 cost: ~300 GPU·hr (§6.6 计入)

输出: π_0 = SFT warm-up checkpoint
       后续 Phase 1 中所有 RL 更新 KL 锚定到 π_0
```

#### Phase 1 — RL Training Loop

```
对每个 mini-batch 的场景 s ∈ D_train：

  Step 1. Rollout (strategy-conditioned group, contextual GRPO, N = 10)
    for k = 1, ..., 10:
      System prompt: 第 k 个临床策略 (NURSE / VERA / SPIKES / ...)
      π_θ 从该 prompt 采样 (温度 ≥ 0.8)，输出 τ_k = [(T_{k,t}, R_{k,t})]
      与 DemMA 多轮交替: DemMA 接收 <response> 进入 patient 状态生成
      记录每 turn 的 (u^p_{k,t}, ℓ_{k,t})

  Step 2. 安全检测
    C_cat(τ_k)        ← 4 谓词多轮 pattern P1–P4 (谓词在 <response> 上抽)
    c_safety(τ_k)     ← safety judge (checklist max, judge call #1)

  Step 3. 性能打分
    R_goal, R_fit, u_terminal ← trajectory judge (合并 1 调用/轨迹, judge call #2,
                                                   输入完整 <think>+<response> 对话)
    D_t, R_t                  ← inline annotation 经临床量表 ordinal tier 抽取 (§4.3)
    r_distress,k,t            = D_{t-1} − D_t              (零额外 LLM 调用)
    r_resistance,k,t          = b_t · (R_{t-1} − R_t)      (零额外 LLM 调用; b_t rule-based)

  Step 4. GDPO 多时域归一化 (§4.6.1)
    R̂_goal, R̂_fit, û_terminal     ← CRank in G_s
    r̂_distress, r̂_resistance       ← percentile rank in G_{s,t} (mask b_t=0 视为缺失)

    A_traj_k         ← R̂_goal,k + R̂_fit,k + û_terminal,k
    A_turn_{k,t}     ← r̂_distress,k,t + r̂_resistance,k,t

  Step 5. α-Weighted dual-horizon advantage (§4.6.2)
    A_{k,t} = 1[C_cat(τ_k)=0] · (A_traj_k + α · A_turn_{k,t})
              − λ_safety · c_safety(τ_k)
    α = 1 default; A8 ablation 做 α sweep ∈ {0.25, 0.5, 1, 2, 4}
    Apply A_{k,t} 到该 turn 的所有 (think + response) token

  Step 6. PPO-clip 策略更新
    标准 PPO-clip 目标，token-level advantage = A_{k,t} for tokens at turn t
    XML format token (<think>, </think>, <response>, </response>) 不参与梯度
    KL 锚定到 π_0 (SFT warm-up checkpoint)，惩罚强度 β_KL
    Lagrangian λ_safety 由 dual ascent 缓更新（每 K 步）
```

每场景共 **2 次 LLM judge 调用**（c_safety + 合并的 R_goal / R_fit / u_terminal），group = 10 → **每场景 20 次 judge 调用**，对比 MAPO 的 90 次 / 场景仍是 4.5× 节省。

**推理阶段**（Phase 1 完成后）：strategy prompt 移除；agent 在与 DemMA（或部署环境）交互时直接面对对话历史**自由生成** `<think>+<response>` 两层结构，已通过 RL 内化"何时倾向哪种策略"的 context-aware 选择能力。这一 train/inference distribution shift 是 standard policy distillation 实践，与 RLVER / MAPO 的 simulator-conditioned training + free inference 一致。

---

## §5 与已有工作的关系

我们沿 6 个**可代码级或协议级独立审计**的维度对比 EC-MTRL 与最相关的 7 个工作。每一列对应一个具体的 design choice，无主观词；任何第三方都可以通过阅读论文与代码验证表中条目。

| 方法 | 轨迹级奖励 | 轮级奖励来源 | 轮级 non-anticipative | 轮级边际成本 | 多维 reward 解耦 | 安全约束分离 |
|---|---|---|---|---|---|---|
| **GRPO** (DeepSeek 24) | 终端单值 | — | — | — | 否 | 否 |
| **MAPO** (arXiv 2603.06194) | batch 归一化 | 外部 LLM-judge per-turn | 否（judge 见全对话） | `O(N·T)` LLM 调用 | 否 | 否 |
| **MT-GRPO** (Zeng et al.) | GRPO 风格 | turn-level advantage（仍依赖外部 reward 在 turn 上的可获得性） | 视外部 reward 而定 | 视来源 | 否 | 否 |
| **GiGPO** (NeurIPS 25) | episode 组内相对 | step 组内相对（anchor state 分组） | 是 | 0 | 否 | 否 |
| **RLVER** (ICLR 26, Tencent) | simulator 输出的情绪标量 | — | — | 0（需改造 simulator） | 单维 | 否 |
| **GDPO** (NVIDIA, arXiv 2026.01) | per-reward 独立归一化 | — | — | — | **是** | 否 |
| **DuCA / HIAN** (arXiv 2603.01481) | session 转化率 | 手工规则轮级 | 是 | 0 | 否（独立归一化 dense+sparse） | 否 |
| **Safe RLHF** (ICLR 24) | helpfulness | — | — | — | helpful/harmless 解耦 | **轨迹级 Lagrangian** |
| **EC-MTRL（本文）** | **R_goal + R_fit + u_terminal（3 head 共用 1 judge call, per-reward CRank）** | **inline annotation → OERS/PAINAD/RTC ordinal tier → state delta → percentile rank** | **是（structurally）** | **0**（轮级通道；c_safety judge 与 C_cat 谓词抽取独立计入） | **是（双时域 + 多 head per-reward 独立归一化）** | **是（C_cat 4 谓词多轮 pattern + c_safety Lagrangian）** |

### 5.1 vs RLVER：最相关，但 4 个维度上做了窄化扩展

RLVER 是 signal-source 思路上**最接近本文**的工作——它同样利用 simulator 自身产出的副信号（情绪标量）作为 reward。本文相对 RLVER 的 delta 是 4 个工程维度的窄化扩展：(i) **副输出获取**——RLVER 需改造 simulator 让其输出连续情绪分（额外 prompt 工程或微调）；本文利用 DemMA 在原论文中**已自然存在**的 environment-native feedback（离散 action labels），不改造 simulator；(ii) **信号粒度**——单维 → 多模态多维（motion + facial + sound）；(iii) **映射方式**——直接连续标量输出 → 临床量表 ordinal tier + state delta + percentile rank（`f` 作 underlying clinical anchoring 但不直接送入训练，§4.3）；(iv) **时域**——仅轨迹级 → 双时域（轨迹级 R_goal / R_fit / u_terminal + 轮级 r_distress / r_resistance）。这一组合不是简单"做更多"，而是把 signal-source 思路推到一个**可被领域量表 principled 锚定**且**轮级可用**的实例化点上——这是把 RLVER 从单一情绪标量推广到 safety-critical multi-modal dialog setting 的最小工程改动。

### 5.2 vs MAPO / MT-GRPO：外部 LLM-judge 路线 vs Inline Annotation 路线

MAPO 与 MT-GRPO 代表"用同一 base LLM 做外部 per-turn judge"这一主流轮级 credit 路线。它们的优势是 judge 看到完整对话后能给出语义复杂的判决；其代价是 (a) `O(N·T)` 额外 LLM 调用——在 group=10、`T≈9` 下意味着每个场景多 90 次 judge 调用；(b) judge 看到 turn `t` 之后的所有 turn → 非 non-anticipative，存在事后归因偏差。**EC-MTRL 不主张 inline 比 external judge "更准"**——这是一个无法在 LLM-simulator 设定下被无监督回答的问题。我们在 §6 设计 *signal-source 匹配算力对照实验*（baseline #4 MAPO + #6 MT-GRPO），让两条路线在同一 testbed、相同 base model、匹配训练 GPU·hr 的条件下直接比较 task quality 与 hacking signature——结果（无论 inline 赢、平、还是输）作为本文的核心实证产出。

### 5.3 vs GDPO / DuCA / Safe RLHF：组件来源与 Integration Novelty

EC-MTRL 的核心机制都有**单独**的相邻工作：(a) per-reward 独立归一化来自 **GDPO**（NVIDIA arXiv 2026.01；但 GDPO 是 single-turn 终端奖励——本文 GDPO 扩展 ① **Multi-horizon GDPO** 把它推广到双时域）；(b) ordinal-aware rank-based advantage 估计承袭 **GOPO**（arXiv 2026.02），本文的轮级 percentile rank in `G_{s,t}` 与 GOPO 在 ordinal reward 上 rank 优于 z-score 的实证结论一致；(c) 轨迹级 + 轮级独立归一化后直接相加来自 **DuCA / HIAN**（arXiv 2603.01481；但 DuCA 的轮级是手工规则，未涉及多维解耦或安全约束；本文 GDPO 扩展 ② **α-weighted dual-horizon advantage** 显式 ablation α 灵敏度，并把"直接相加 OK"从 community consensus 升级为 PBRS 定理保证）；(d) 安全与性能分离来自 **Safe RLHF**（ICLR 2024；但 Safe RLHF 是 single-turn）；(e) **HiPER** (NeurIPS arXiv 2602.16165) 启发了"按推理层次分配 credit"的概念框架——但 mechanism differs：HiPER 在 hierarchical policy 的 high-level planner / low-level executor 两个 sub-policy 间分配 credit，使用两层 actor 结构；本文是 **single policy + α-weighted advantage**（不引入 hierarchical policy / token-routing 复杂度），仅借鉴"轮级 vs 轨迹级 credit 显式区分"的 conceptual 启发；(f) R_fit 第 4 项 *epistemic discipline* 子项的设计直接对应 **RubRIX**（arXiv 2601.13235）的 Uncritical Affirmation / Epistemic Arrogance 风险维度；(g) `r_resistance` 的 care-bid mask 形式对应 dialogue RL 中的 **multi-stage gated reward function**（含 Step-by-Step Task-Oriented Dialog RL，arXiv 2406.14457）；(h) `<think>+<response>` 两层输出格式承接 **R1**（reasoning trace as explicit token surface）的标准实践；(i) §4.3 的 state-delta 轮级 reward 形式由 **Potential-Based Reward Shaping** 的经典定理保证 policy invariance（**Ng et al. 1999** vanilla PBRS；**Wiewiora et al. 2003** action-dependent advice 扩展，覆盖 care-bid mask；**CURIO arXiv 2504.03206** 在 LLM 域近期应用），本文是首次把 PBRS framing **显式应用到 LLM-simulator 多轮对话 RL 的 inline-annotation 派生 reward 上**，这把 α-weighted dual-horizon advantage 的 policy invariance 从 community consensus 升级为经典 RL 定理；(j) GDPO 扩展 ③ **Strategy-conditioned group as contextual GRPO with structured strategy prior**——10 临床策略 system prompt × 各 1 sample 把 GDPO group sampling 从 standard GRPO 推广到 contextual GRPO，让 group 内的 trajectory 多样性源于 principled 行为对照而非 sampling noise，与 dialogue policy 文献中的 structured advice / action-space prior 思想一致。本文的 method-level 贡献是 **GDPO 的三个扩展（Multi-horizon + α-weighted dual-horizon + Strategy-conditioned group）+ POMDP framing + PBRS framing + integration**——三个扩展中前两个是相对 GDPO 原版的非平凡 method-level 推广，第三个是 GDPO group sampling 的 contextual 推广；其余组件 (a)/(c)/(d)/(f)/(g)/(h) 与 inline annotation 轮级通道、u_terminal 末段 LLM-judge rubric、`<think>+<response>` 两层 reasoning surface 同时整合到一个多轮训练目标中。其价值是否成立，最终取决于 §6 的实证结果——若 integration 在匹配算力下显著优于任何单独使用其中一两个机制的 baseline，则 integration 本身的工程含金量就成立。

---

## §6 实验计划

### 6.1 五个研究问题与 Pre-Registered 判定阈值

我们在投稿前**预先注册** 5 个研究问题与各自的判定阈值（"支持 / 不支持"分界线在实验启动前锁定，所有判定逻辑公开发布）。每条 Q 同时声明实验失败时的 fallback narrative，避免 "结果不利就重写 framing" 这一可被审稿人质疑的常见做法。

| Q | 研究问题 | Primary Metric | 支持阈值 | 不支持阈值 | 实验失败的 fallback narrative |
|---|---|---|---|---|---|
| **Q1** | 在匹配算力下，inline annotation 轮级通道相对**外部 LLM-judge 路线**（MAPO + MT-GRPO）是否在 task quality 上不劣且 cost 显著低？ | 人类 pairwise preference + 训练 GPU·hr | win-rate ≥ 50%（95% CI 下界 > 45%） **且** GPU·hr ≤ MAPO 的 0.3× | win-rate ≤ 45% **或** GPU·hr 节省 < 2× | claim 收缩为 "matched quality at structurally lower cost"；放弃 quality-superior 表述；该结果仍构成对 LLM-simulator RL 设计选择的有用 negative-on-judges 经验结论 |
| **Q2** | **双时域融合**（轨迹级 + 轮级）相对单时域是否更好？ | 评估 judge 的 (R_goal + R_fit + u_terminal) 加总 + 人评 preference | Full > {纯轨迹级, 纯轮级} 均 ≥ 3 pp（paired bootstrap p < 0.05） | 与最强单时域差距 < 1 pp | 撤回 "双时域必要" claim；integration novelty 收缩为 "signal-source + 安全约束"两项 |
| **Q3** | (a) **Strategy-conditioned group**（10 临床策略 prompt × 各 1）相对 unified prompt 同分布采样 N=10 是否在 final policy 质量与训练 variance 上更好？(b) **α-weighted dual-horizon advantage** 在 α sweep 下是否找到稳定区间（α=1 ± k 范围内 final policy 质量不显著退化）？ | (a) 训练曲线 variance + final (R_goal + R_fit) 加总；(b) final (R_goal+R_fit+u_terminal) 在 α ∈ {0.25, 0.5, 1, 2, 4} 上的曲线 | (a) variance ≤ 0.7× **且** final R 不劣（CI 不显著低于 baseline）；(b) α ∈ [0.5, 2] 区间内 final R 与 α=1 差距 < 2 pp | (a) final R 显著低于 unified prompt baseline；(b) α 在 [0.5, 2] 内任意点 final R 显著退化（> 5 pp） | (a) 撤回 GDPO 扩展 ③（Strategy-conditioned group），回退到 standard GRPO unified prompt；(b) 撤回 α=1 默认，根据 sweep 结果推荐另一个 α 值或承认 α 敏感性是 method limitation |
| **Q4** | 两层安全约束相对单层 / 无安全约束的灾难违规率？ | C_cat 违规率（人类判定 + 规则） | 两层相对无安全 ≥ 50% 下降 | < 20% 下降 | 安全设计简化为单层 c_safety；§4.5 重写 |
| **Q5** | 跨亚型 / 跨冲突类型的泛化？ | held-out 上 (R_goal + R_fit + u_terminal) + preference 相对训练分布的衰减 | 衰减 ≤ 20% | > 40% | claim 限制到训练分布内；泛化部分移入 §7 局限性 |

**统计协议**：每种方法独立训练 ≥ 3 seeds（主对比 EC-MTRL vs Tier 2 共 4 套均跑 5 seeds）；主指标 mean ± std；pairwise 比较用 paired bootstrap (10,000 resamples) 报 95% CI；Q1–Q5 family 用 Holm-Bonferroni 校正控制 family-wise error rate 0.05。

### 6.2 Baseline 清单（按 Tier 组织）

**Tier 1 — 经典 RL baseline**（必跑，匹配 compute）

| # | Baseline | 配置 |
|---|---|---|
| 1 | **SFT only** | 在 DemMA 配套专家护工对话上做 SFT（数据规模、license、与 testbed scenario overlap 在附录 D 披露），无 RL |
| 2 | **GRPO**（单一终端奖励） | 标准 GRPO，奖励 = (R_goal + R_fit) 加总（轨迹级单值，无 u_terminal、无双时域、无 per-reward 解耦） |
| 3 | **PPO**（单一终端奖励） | 标准 PPO，同上奖励；提供 critic-based vs critic-free 对照 |

**Tier 2 — 同类多轮 RL（公开代码已确认）**

| # | Baseline | 代码 | 与本文差异 |
|---|---|---|---|
| 4 | **MAPO** (Wenke Huang) | `WenkeHuang/MAPO` | turn-level + batch-level mixed advantage；轮级用同 base model 外部 judge per-turn |
| 5 | **MT-GRPO** (Siliang Zeng) | `SiliangZeng/Multi-Turn-RL-Agent` | GRPO + turn-level advantage estimation；最直接的 turn-level credit baseline |
| 6 | **RLVER** (Tencent) | `Tencent/digitalhuman/RLVER` | simulator-side reward；单维情绪 + 仅轨迹级 |

**Tier 3 — 人类锚点**

| # | Reference | 配置 |
|---|---|---|
| 7 | **人类护工 reference** | 不训练。从 DemMA 配套对话中抽 ≥ 100 对 matched-scenario "人类护工 vs EC-MTRL"，做 blind pairwise preference。**用途**：锚定 EC-MTRL 与人类专家在 empathy / safety / consistency 各维度上的 gap，是 Use-Inspired 工作必备的"vs human baseline"。**不**作为优化对象。 |

### 6.3 Ablation（method 内组件，独立于 baseline）

| Ablation | 变体 | 验证什么 | 关联 Q |
|---|---|---|---|
| A1 轮级通道开关 | {on (r_distress + r_resistance), off (仅轨迹级 3 head)} | 轮级信号本身的边际贡献 | Q2 |
| A2 轮级信号来源 | {inline annotation tier (本文), 同 base model 外部 LLM judge per-turn, 无} | inline ordinal tier vs external judge 的 head-to-head | Q1 |
| A3 安全层数 | {0, 1 (仅 c_safety), 2 (C_cat 4 谓词 + c_safety)} | 两层设计是否必要 | Q4 |
| A4 轨迹级归一化 | {CRank (本文), z-score} | GOPO 的 rank 归一化是否优于 z-score | — |
| A5 轮级 reward 形式 | {state delta + percentile rank (本文), raw tier value, z-score on delta} | §4.3 / §4.6 的 ordinal-aware 设计是否最优 | — |
| A6 Ordinal tier 边界 | {当前临床锚点边界 (本文), 粗粒度 (3 级 / state), 等距 (4 级均匀切分)} | 主结论对 ordinal tier 边界设计的鲁棒性 | — |
| A7 c_safety 聚合 | {max (本文), sum, mean} | §4.5 的 max 选择是否最优 | Q4 |
| **A8 α-weighted advantage sweep** | α ∈ {0.25, 0.5, 1 (本文 default), 2, 4} | GDPO 扩展 ②（α-weighted dual-horizon advantage）的灵敏度区间；PBRS 不变性下 α 不改变最优策略，但样本效率随 α 变化 | Q3 (b) |
| **A9 Strategy-conditioned group** | {10 策略各采 1 (本文), unified prompt 同分布采 N=10 (standard GRPO)} | GDPO 扩展 ③（Strategy-conditioned group as contextual GRPO）相对 standard GRPO 的边际贡献 | Q3 (a) |
| **A10 Care-bid mask** | {on (RTC-aligned gating, 本文), off (r_resistance 在所有 turn 计算)} | RTC-aligned reward gating 是否减少 zigzag hacking | Q1-hack |
| **A11 u_terminal 来源** | {LLM judge on last 3 turns (本文), 直接聚合最后一轮 (D_T, R_T)} | LLM judge 末段评估 vs turn-state 聚合的相对优势（last-turn noise robustness） | Q2 / Q1-hack |
| **A12 Warm-up 充分性** | {1 epoch SFT, 2 epoch SFT, 无 warm-up（直接 RL）} | SFT warm-up 对 `<think>+<response>` 输出格式 stability 与 RL 收敛性的影响 | — |

### 6.4 评估指标与人评协议

**Primary metric**：**blind pairwise human preference**（≥ 200 对 dialog × ≥ 3 raters）。这是 NeurIPS 多轮 dialog RL 论文的当前社区共识 primary metric——LLM-as-judge 即便分离训练 / 评估也仍存在已知的 systematic bias（自我偏好、长度偏好等），人评是唯一不可被 judge-side hacking 攻击的对照。

**Secondary metrics**（自动化）：
- **Held-out evaluation judge** 的 R_goal / R_fit / u_terminal 三个 reward 分数（评估 judge 与训练 judge 的 base model family 严格分离，§4.4）；
- C_cat 违规率（规则触发 + 人类抽样验证）；
- c_safety 均值与高 c_safety (≥ 1) 比例；
- 输出 n-gram diversity（用于 §6.5 hacking audit）；
- 训练 GPU·hr 与 inference token cost。

**Rater 招募与 IRB**：
- ≥ 3 名 rater，至少 1 名具备痴呆护理或长程护理培训背景，其余通过 ≥ 2 小时 calibration 训练并在 20 题 sanity 集上达 Cohen's κ ≥ 0.6；
- 人类评估方案在实验启动前向所属机构 IRB 提交（仅涉及成人 rater 评估 LLM 生成的对话，不涉及真实病人）；
- 评估材料中所有姓名、地点 token 替换为占位符；severity-3 场景在评估界面附内容预警；rater 可随时退出。

### 6.5 Reward Hacking Audit（3 层）

DemMA 是 frozen LLM；agent 在 KL 锚定下训练 8–10 轮 × 多场景 × 多 epoch 后，可能学到"诱发 smiling 或抑制 frowning 的特定 phrasing"，使轮级 reward（`r_distress` / `r_resistance`）在不真正改善护理质量的情况下被推高。这是所有 LLM-simulator RL 的共有风险，本文不掩盖，而是配 3 层 audit 公开 report。

| 层 | 探针 | 操作 | 可接受阈值 | 触发撤回的阈值 |
|---|---|---|---|---|
| **L1** N-gram drift | 训练中护工侧高频 n-gram 分布的漂移 | 每 K 步采样 M 条 rollout，统计 top-K n-gram 的 TVD vs SFT reference | TVD ≤ 0.4 | TVD > 0.6 → caution；伴随 hacking pattern 时撤回 |
| **L2** Simulator perturbation | 用 DemMA 的变体（不同 random seed / system prompt phrasing / LoRA adapter）替换原 DemMA 跑同一 trained policy | 比较替换前后轮级 reward（`r_distress` 与 `r_resistance` 各自平均）均值 | 跌幅 ≤ 15% | **跌幅 > 30% → 触发 §7.1 fallback narrative**：撤回 inline annotation 优势 claim，主结论改写为 "inline annotation 在 frozen LLM simulator 下可能 over-fit 到 simulator 特定 token 分布" 的 negative finding。**理论动机**：CURIO (arXiv 2504.03206) §5 实证 non-PBRS reward 在 LLM RL 下出现 length inflation 等 hacking——本文 §4.3 的 PBRS shaping 形式从理论上规避此类 hacking；本 audit 检验 PBRS 的 policy invariance 在 frozen LLM simulator 这个具体 setting 下是否仍然 holds（即 proxy potential `φ_D = −D_t` 与 oracle potential `φ^* = −\text{distress}(z_t)` 的 alignment 是否在 trained policy 下不破坏）。 |
| **L3** 高轮级 / 低轨迹级 reward 人评抽样 | 抽 ≥ 50 条 "高 (r_distress + r_resistance) 累积改善 / 中低 (R_goal + R_fit)" 的 rollout，由 ≥ 2 名人类 rater 独立判断是否存在 hacking 迹象 | 报告 hacking-rate（双盲 majority vote） | hacking-rate ≤ 20% | **> 30% → 触发撤回**，整篇 framing 重写为 audit-focused negative finding |

**关键诚实性**：3 层 audit 的结果（无论好坏）均作为本文**核心实证产出**之一公开 report。我们不宣称 EC-MTRL 免疫 reward hacking；我们宣称的是 hacking 程度在标准 KL 约束下**可被量化、可被 falsify**。

### 6.6 Compute Budget（Fermi 估算）

| 条目 | 单次 (GPU·hr, A100-80GB) | 数量 | 合计 |
|---|---|---|---|
| **Phase 0 — SFT warm-up**（一次性，§4.7；DemMA-native data 直接抽取，无 GPT-4o 标注） | ≈ 300 | 1 | **300** |
| EC-MTRL 主训练（Phase 1 RL：rollout + PPO，含 strategy-conditioned group=10 + α-weighted advantage） | ≈ 80 | 5 seeds | 400 |
| Tier 1 baselines (#1–#3) | ≈ 50 | 3 baseline × 3 seeds | 450 |
| Tier 2 baselines (#4–#6) | ≈ 100 | 3 baseline × 5 seeds | 1,500 |
| Ablation A1–A12（含 A8 α sweep / A9 strategy-conditioned group / A10 care-bid mask / A11 u_terminal 来源 / A12 warm-up 充分性；A8 sweep 5 个 α 值各 3 seeds） | ≈ 60 | 12 ablation × 3 seeds + A8 额外 4 个 α × 3 seeds | 2,400 |
| Evaluation（held-out judge + 人评准备） | — | — | 200 |
| Hacking audit（含 simulator perturbation 重跑） | — | — | 250 |
| **合计目标预算** | | | **≈ 5,500 GPU·hr** |

按 8×A100 持续可用估算 wall-clock ≈ **4.5 周**；DemMA-native SFT warm-up 数据准备约 0.5 周（vs zh_8 / E 方案需要 GPT-4o 标注 + 临床 RA 内容审核 ~2-3 周），加人评筹备、IRB 审批与场景库构造，**端到端 6–7 周**。

**最小可发表子集**（≈ 2,100 GPU·hr，单条波动 ±30%）：

- **Phase 0 SFT warm-up（不可砍，`<think>+<response>` 两层输出格式的基础）**；
- Baselines：#1, #2, #4（核心 Q1 对照）, **#7（人类 reference）**；
- Ablation：A1 (轮级通道), A3 (安全层数), A8 (α sweep, 至少 α ∈ {0.5, 1, 2} 三档), A9 (strategy-conditioned group), A12 (warm-up 充分性) — 对应 Q2 / Q3 / Q4 主线；
- Audit：3 层全跑（不可砍，关乎 Q1-hack 撤回判定）；
- 人评 ≥ 100 对（不可砍）；
- Cross-subtype generalization (Q5)。

**可后补**（rebuttal 阶段或 final version）：Tier 2 #5 #6、A4 / A5 / A6 / A7 / A10 / A11 fine-grained ablation、A8 α sweep 的 {0.25, 4} 极端值。

---

## §7 风险与局限

### 7.1 三大风险及 fallback narrative

我们显式列出本 proposal 最可能失败的 3 个原因，每条对应 §6 的具体 audit / 实验，并给出失败时的 honest fallback。**这不是"为失败找借口"，而是把 framing 的健壮性写在前面**——让 reviewer 知道作者已认真思考过可能失败的路径。

**风险 1（最致命）：DemMA action label ↔ 真实病人福祉的 hidden correlation 不足以支撑 RL 信号**。`r_distress` 与 `r_resistance`（由 `ℓ_t` 经临床量表 ordinal tier + state delta 派生）作为 RL 信号驱动 policy 时，policy 在数学上是在最大化 DemMA 条件标签生成分布所对应的 patient-state tier 改善，而非真实病人福祉。若 DemMA 的标签生成存在系统性偏置（例如对某种话术总倾向输出 smiling），policy 会被锁死到该话术。**识别方式**：§6.5 L2 simulator perturbation（轮级 reward 跌幅 > 30%）+ Tier 3 人类护工 reference（若 EC-MTRL 在某些维度偏离人类专家而非接近，是间接证据）。**Fallback**：core claim 退为 "inline annotation 在 frozen LLM simulator 下作为 RL signal 可能 over-fit 到 simulator 特定 token 分布" 这一 negative-on-inline 经验结论；contribution 类型从 Use-Inspired 偏向 Negative Results，但仍构成对 LLM-simulator RL 设计选择的有用社区警示。

**风险 2：EC-MTRL 在匹配算力下与 MAPO/MT-GRPO 的 task quality 落入不确定区间（45% < win-rate < 55%）**。LLM-judge 当前已足够强；inline 通道可能并不显著优。**识别方式**：§6.1 Q1 的 pairwise preference + CI。**Fallback**：claim 严格收缩为 "matched quality at structurally lower cost"（GPU·hr ≤ 0.3×），把 §Q2 双时域、§Q3 策略 group、§Q4 安全两层、§Q5 泛化的累计证据组合作为 integration novelty 的支撑。这种结果对 NeurIPS main track 的接收概率低于 quality-superior 情形，但仍在 borderline 之上。

**风险 3：C_cat 谓词抽取分类器的 precision/recall 不达临床可接受门槛**。若 precision 低 → 硬否决错杀好轨迹 → 训练崩溃或 over-cautious policy；若 recall 低 → 灾难违规漏检 → §Q4 数字失真。**识别方式**：附录 D audit 的 precision ≥ 0.85 / recall ≥ 0.75 / Cohen's κ ≥ 0.7（临床研究常用门槛）。**Fallback**：未达门槛的子规则降级为 c_safety 子项而非硬否决；C_cat 简化到最高确定性的"高风险错误事实确认 + 明确虐待"两类，其余降级；§4.5 安全两层的 framing 据实修订为"两层 + 部分降级规则"。

### 7.2 局限性

1. **Signal-source 依赖**：本方法的 setting 前提是 simulator 已能产出 *environment-native inline feedback*（有限字母表的离散标注）；对不具备该属性的 simulator，方法退化到 RLVER 路径（"改造 simulator 让它输出"）。Environment-Coherent 作为 design axis 的实际 reach 受 simulator 生态限制。
2. **DemMA 单点依赖**：Primary testbed 与 DemMA 可用性强绑定。我们公开 DemMA 的 minimal adapter API 与 frozen checkpoint hash，但若 DemMA 不可用，痴呆护理实验无法在第三方独立复现（方法论本身仍可在任意带 inline annotation 的 simulator 上复现）。第二 testbed 占位见附录 D。
3. **临床量表的简化**：OERS/PAINAD/RTC 的 18 项查表是临床量表的工程简化；临床医生可能合理质疑这种"压平"丢失了"受训观察者 + context"的判断。§6.3 A6 ablation 中的 learned weights 作为对此的 sensitivity check。
4. **无 ground truth**：`ℓ_t` 不是真实病人行为观测；DemMA 条件生成与真实病人福祉之间的相关性**无法被无监督验证**。本文的 claim 完全建立在三条 *可审计的工程性质* 上（inline commitment / zero marginal cost / clinical-scale mappability），不在真值上。
5. **从本文到临床部署的 gap**：本工作不主张训练得到的 agent 适合直接临床部署；至少需要 IRB、临床试验、监管审批、医生在回路监督等多个独立环节。详见 §8。
6. **Compute scale**：≈ 5,500 GPU·hr 为学术实验室量级；工业级 scale（更大 policy 模型、更长 context）下的外推**不在本文 claim 范围**。
7. **语言/文化范围**：本工作所有实验在英文 dialog 上进行；跨语言 / 跨文化的 dementia care 表达差异不在本文范围。
8. **`<think>+<response>` 输出格式依赖 SFT warm-up**：两层 R1 风格输出格式需要 Phase 0 SFT warm-up（§4.7）使 base LLM 先学会稳定输出 XML 结构；warm-up 不充分时 RL 早期可能出现格式 drift。本工作的 warm-up 数据**直接抽取自 DemMA 配套对话的 caregiver 双侧 trace**（§4.7 / 附录 B.5），无 GPT-4o 标注成本；§6.3 A12 ablation 检验 warm-up 充分性，最小可发表子集要求 warm-up checkpoint 输出格式 success rate ≥ 95%。

---

## §8 Broader Impact & Ethics

本工作涉及 safety-critical 医疗对话场景，存在若干必须明确处理的伦理与社会影响维度。

**部署边界**。本工作**不主张**训练得到的 caregiver agent 直接用于真实痴呆病人交互。任何从本工作到临床应用之间至少需要：(a) 独立 IRB 与临床试验、(b) 临床医生在回路实时监督、(c) 针对 OOD persona / 罕见亚型 / 共病 / 文化差异的独立安全评估、(d) 监管机构（FDA / CE 或对应地区）医疗 AI 审批。本 proposal 的直接贡献仅在方法学层面。我们在 release 时附 *responsible use statement*，明确声明 artifact **不得**被以未经独立审批的方式部署于真实病人交互。

**Therapeutic Fibbing 的伦理立场**。Therapeutic Fibbing（对认知障碍患者的善意非真实陈述）作为 §3 的 10 策略之一被纳入训练 group，是因为它**已被广泛临床文献描述**（Day et al. 2011 报告 ≈96% 护工实践中使用过）且在 severity-3 场景下减轻患者痛苦的证据存在。我们的立场是：该策略的使用**不是本文的主张**，而是对真实护工策略空间的描述；§6.6 emergent analysis 中若 trained agent 在某些场景下涌现该策略，**不**作为正向结论解读。临床社区对该策略仍存在合理的伦理分歧，本工作尊重该分歧并不替读者作价值判断。

**Bias 与代表性**。(a) DemMA 训练 / 验证数据的人种 / 性别 / 文化代表性按 DemMA 原论文披露（附录 D）；若 DemMA 有偏，本 agent 继承该偏差——这是 §7.2 局限 7 的具体来源。(b) 训练 judge 与评估 judge 的 base model 均可能携带 emotion / politeness / culture bias，会传导至 reward；§4.4 与 §6.4 的 judge 分离协议部分缓解，但不能消除。(c) 10 策略中 NURSE / VERA / SPIKES / DICE 主要源自北美 / 英国临床文献，其他文化背景的护理规范可能不同。

**Dual-use 与误用风险**。同类多轮 RL 方法在原理上**可被用于**训练面向脆弱人群（老年人 / 认知障碍者 / 儿童）的操纵性 agent——例如以情绪信号作为 reward 的诱导性销售对话。我们的应对：(a) release artifact 时附明确 responsible use statement；(b) testbed 的 C_cat 规则集与 safety rubric 一并 release，为后续使用者提供 reference safety scaffold，降低从零构造时遗漏安全约束的风险。

**LLM 模拟脆弱人群的本体论争议**。对痴呆病人做 LLM 模拟并在其上做 RL 训练，本身属于 ML & ethics 近年的开放议题（参考 ML4H 相关讨论）。我们承认 "用 LLM 代表痴呆患者" 在认识论上有争议，已在 §7.2 局限 4 显式承认；本工作选择推进的理由是 **simulator-based RL 是当前唯一可在不接触真实脆弱人群的前提下迭代探索 caregiver agent 设计空间的工程路径**——这是一个 trade-off，不是无瑕方案。

**人类评估伦理**。已在 §6.4 列出：rater 招募门槛、补偿、IRB protocol、de-identification、知情同意、severity-3 场景内容预警、随时退出机制。

**资源 / 碳成本**。≈ 4,250 GPU·hr 估算产生约 0.5–1.5 吨 CO2eq（按所在数据中心碳强度，具体在终稿 report）；最小可发表子集 ≈ 1,800 GPU·hr 约为一半。我们承诺在 final version 中 report 实际数字，且在选择 base model 与 batch size 时优先考虑 efficiency。

**本节分析自身的局限**。本节 Broader Impact 分析仅基于 ML 研究者视角，**未做正式的 stakeholder consultation**——我们没有系统访谈临床医生、护工、痴呆病人家属或患者本人。这是分析本身的局限。在 final paper 阶段，我们计划与一名痴呆护理临床伦理学家做至少一轮独立 review，并将其反馈纳入 ethics statement 修订。

---

## 附录 A：10 临床策略 System Prompt 模板

本附录给出 §3.2 表中 10 个策略的完整 system prompt 模板（每条 80–150 token），用作 §4.2 rollout 阶段的策略条件化采样。所有模板在训练前 lock，全程不修改；模板版本号 + hash 与训练 checkpoint 一并 release。下面给出策略 1（NURSE）作为示例：

> **NURSE 策略 system prompt 模板（示例）**：
> 你是一名痴呆护理护工，按 NURSE 共情沟通框架（Back et al., 2005）行事。在与患者的对话中，每一轮按以下顺序优先：(1) Naming——为患者表达的情绪命名（"听起来您有点担心"）；(2) Understanding——表达理解；(3) Respecting——表达对患者的尊重；(4) Supporting——表达陪伴与支持；(5) Exploring——温和探索患者需要。**暂时搁置事实纠正**，先处理情绪。每轮回复保持 30–80 字，自然口语化。

其余 9 个策略（VERA / SPIKES / DICE / Reality Orientation / Therapeutic Fibbing / Reminiscence Therapy / Montessori / Redirection / Non-committal）的完整模板列于本附录的 9 个独立子节，结构与上例一致：(策略名 + 临床来源 + 操作步骤 + 对事实冲突的姿态 + 风格约束)。每条模板的完整文本 ≤ 200 字，便于 reviewer 逐条审计。

---

## 附录 B：完整 Rubric 与 OERS 查表的临床引用

**B.1 R_goal / R_fit / u_terminal 三个 trajectory rubric 完整 prompt**。本附录给出 §4.4 中 R_goal（4 项 checklist）、R_fit（4 项 checklist 含 epistemic discipline）、u_terminal（rubric 评末段 3 轮的 distress + resistance 二维）的完整 prompt：每项含 0/1/2 三档锚点描述、2 条 few-shot 样例（一个 conflict-resolved + 一个 distress-escalating）、以及 evidence-anchoring 指令（要求 judge 在打分前引用对话中的具体 turn 作为依据，遵循 RULERS arXiv 2025.01 协议）。三个 rubric 的全文版本号 + hash 与 training judge checkpoint 一并 release。

**B.2 c_safety 子项 checklist 完整 rubric**。同样结构：3 类担忧（反复直接矛盾 / distress 升高后继续升级 / 婴儿化、否定、居高临下）各含 0/1/2 锚点 + few-shot 样例。

**B.3 OERS / PAINAD / RTC 完整 ordinal tier 边界规则**。§4.3 给出了 D_t（distress）与 R_t（resistance）4 级 ordinal tier 的核心边界示例；本附录补完整 18 个 inline annotation label 在 OERS（6 dimensions × intensity levels）/ PAINAD（5 items × 0/1/2）/ RTC（13 items × frequency）中的具体 anchor 文本，以及完整 tier 映射规则（含 disjunctive rule 全集与同 tier 内行为锚点的语义边界判定）。

**B.4 C_cat 规则全集与谓词定义**。§4.5 描述了 4 谓词多轮 pattern P1–P4；本附录给出每条规则对应的 stance / permit / hard_correct / coercive 谓词的精确 if-then 定义、训练判据样例、谓词抽取作用域（仅在 `<response>` token 上抽，不读 `<think>`）、以及附录 C calibration study 中将报告的 precision / recall / κ 目标门槛。

**B.5 Phase 0 SFT Warm-up Data Preparation Protocol（DemMA-Native）**。§4.7 Phase 0 引入 SFT warm-up 让 base LLM 学会输出 `<think>+<response>` 两层格式。**关键工程便利**：DemMA 配套对话数据本身已含 caregiver 侧的 `<think>` 与 `<response>` 双侧 trace（DemMA 论文为训练 patient simulator 学习"在 caregiver 这么 think 时如何反应"而保留），可**直接抽取**作为 SFT 数据，**无需 GPT-4o 标注、无需临床 RA 审核内容质量**。本附录详述：
- **数据来源**：DemMA 配套 dementia care 对话集合（附录 D，~3–5k dialog；含完整 caregiver `<think>+<response>` + patient (utterance, action labels) 双侧 trace；license 与 testbed scenario overlap 状态在附录 D 披露）；
- **抽取流程**：(1) 用 regex / XML parser 从 DemMA dialog 中提取 caregiver 侧的 `<think>` 与 `<response>` 段，patient 侧（utterance + action labels）作为对话 context；(2) 每个抽取出的 (think, response) 对随机配一个 strategy prompt（10 临床策略中均匀抽取）作为 SFT 训练样本的 system prompt；(3) 抽取脚本通过 unit test + few-shot manual check 验证（XML boundary 解析、context-target 对齐），属于工程标准实践——**内容质量已由 DemMA 论文的临床专家 validation 保证，无需独立 RA review**；
- **SFT 训练规格**：base model = Qwen2.5-7B-Instruct（或所选 base）；1–2 epoch，标准 cross-entropy on `<think>+<response>` target tokens；学习率 5e-6；batch size 32；估算 ≈ 300 GPU·hr；
- **Validation 门槛**：warm-up checkpoint π₀ 在 held-out validation set 上输出 `<think>+<response>` 两层 XML 格式 success rate ≥ 95%（自动 parse 验证：标签完整、嵌套正确、内容非空）；未达门槛时增加 SFT epoch 或扩大数据量；
- **工作量对比**：DemMA-native 抽取 ~0.5 周 wall-clock；vs zh_8 / 早期方案需要 GPT-4o 标注 + 临床 RA 内容审核 ~2-3 周 wall-clock；
- **失败模式 honest 披露**：DemMA caregiver trace 的 reasoning style 是 DemMA 论文作者团队的标注约定，可能与其他 dementia care 临床实践有 stylistic 偏差；§6.3 A12 ablation（warm-up 充分性）会检验该 bias 对 RL 收敛性的实际影响；§7.2 局限 8 已明确标注此依赖。

---

## 附录 C：Preliminary Calibration Study Protocol（投稿前 commit milestone）

为缓解 §7.1 风险 1（DemMA action label ↔ 真实病人福祉的 hidden assumption）与 §7.1 风险 3（C_cat 规则可靠性），我们在投稿前承诺完成两项 preliminary calibration study：

**C.1 DemMA label ↔ OERS expert annotation alignment study**

- **样本**：随机抽 K = 500 个 DemMA 生成的 (病人 utterance, action label) 对，覆盖 9 个亚型 × 5 个冲突类型 × 3 级 severity 分层抽样；
- **协议**：≥ 2 名具备痴呆护理评估培训的临床专家独立按 OERS / PAINAD 量表对每个 utterance 打 ground-truth 标签（双盲）；
- **指标**：DemMA action label 与专家 ground truth 的 Cohen's κ；分维度 report（Pleasure / Anxiety / Sadness / Anger / Interest / Engagement 各一）；
- **可继续门槛**：bootstrap (1000 resamples) 95% CI 下界 ≥ 0.4（医学研究中 fair agreement 的常用阈值）；未达门槛则在投稿正文中显式 report 该数字、撤回 §1.4 属性 3 的"可被领域量表 principled 锚定"表述、并把 `f` 重新定位为"轻量校准函数"而非"临床量表绑定";
- **执行窗口**：投稿前 4–6 周；专家招募与补偿独立 IRB 报备。

**C.2 C_cat 谓词抽取 audit**

- **样本**：≥ 200 条 C_cat 规则触发的 turn-pair（precision audit）+ ≥ 200 条人类 rater 标注为"高风险护工话术"的 turn-pair（recall audit）；
- **指标**：precision / recall / Cohen's κ；
- **门槛**：precision ≥ 0.85, recall ≥ 0.75, κ ≥ 0.7（临床研究常用门槛，§7.1 风险 3）；
- **未达门槛时**：未达门槛的子规则降级为 c_safety 子项而非硬否决，§4.5 安全两层 framing 据实修订。

两项 study 的结果（无论达标与否）均作为本工作的 *committed milestone*，与主实验数字一同 report；C.1 的具体数字将出现在投稿正文 §6 开头的 "Preliminary evidence" 段落中。

---

## 附录 D：DemMA 数据披露与第二 Testbed 占位

**D.1 DemMA 配套 dialog 数据**。本工作 §6.2 baseline #1（SFT only）使用的训练数据来自 DemMA 论文配套发布的专家护工对话集合。我们将在 final version 中披露：(a) 数据规模与分布（亚型 × 冲突类型 × severity）；(b) 数据 license 与可获得性；(c) 与本工作 testbed scenario 的 overlap 状态及 train/test split 协议（按 persona-level split 严格不重叠）。

**D.2 DemMA 9 亚型 / 人种 / 性别 / 年龄分布**。按 DemMA 原论文披露；本附录 cross-reference 该分布并标注 §7.2 局限 7 的具体来源。

**D.3 Frozen DemMA checkpoint hash + minimal adapter API**。我们公开一个 `DemMASimulator` 的 Python 接口（输入：dialogue history + persona；输出：utterance + action_labels JSON），便于复现者对 API 接入而非整套 DemMA 代码。

**D.4 第二 testbed 占位**。若 reviewer 担心 DemMA 单点依赖，我们可在第二 testbed（基于开源 LLM + prompt-engineered inline emotion tag 构造的非临床对话环境）上复现 §6 主对比 E1 + 安全 ablation A3，验证方法**工程可迁移性**——*不*独立验证 §6 主结论。第二 testbed 的具体 base model、tag 字母表、calibration 协议与预算列于附录 D.4 子节，编排为 rebuttal-ready material。

---

## 收官

本 proposal 在 zh_8 基础上做了以下主要修订（按重要性降序）：

1. **方法骨架重设**：明确 EC-MTRL 是 **GDPO 的三个扩展 + POMDP framing + PBRS framing**——(1) **Multi-horizon GDPO**（per-reward 独立归一化推广到 trajectory + turn 双时域）；(2) **α-Weighted dual-horizon advantage**（双时域 normalized advantage 用单标量权重融合，PBRS 不变性保证任意 α > 0 下最优策略集合不变）；(3) **Strategy-conditioned group as contextual GRPO**（10 临床策略 system prompt × 各 1 sample，把 GDPO group sampling 推广到 contextual GRPO with structured strategy prior）。Caregiver agent 输出 `<think>+<response>` 两层 R1 风格结构（无 plan）；引入 **Phase 0 SFT warm-up**（直接抽取 DemMA 配套对话的 caregiver trace，零 GPT-4o 成本）；environment-native inline annotation 定位为 environment-native reward signal 的工程基础，POMDP framing 把"不是真值"从免责升级为 first-class anchor；
2. **删除 α 校正全章**（含 BLUE / MVUE 推导）：基于 DuCA / MAPO / GiGPO 的社区共识，独立归一化已足以解决双时域 broadcast 问题，引入 α 是 redundant；
3. **Baseline 结构重设**：按社区习惯分 Tier 1 (GRPO/PPO) + Tier 2 (MAPO/MT-GRPO/RLVER, 均有公开代码) + Tier 3 (人类护工 reference)，"inline vs external judge" 从 baseline 降到 ablation；
4. **5 个 Q + pre-registered 阈值**：每条 Q 同时声明判定阈值与失败时的 fallback narrative，Q3 同时验证 (a) strategy-conditioned group vs unified prompt + (b) α-weighted advantage 在 α sweep 下的稳定区间；
5. **3 层 reward hacking audit + falsifiable 撤回线**：simulator perturbation 跌幅 > 30% 或 hacking-rate > 30% 触发主结论撤回与 framing 改写；
6. **Broader Impact**：新增本节分析自身的局限段，对齐 NeurIPS 2024+ ethics guidance；
7. **删除 zh_8 的 ASP-1..4 编号、two-narrator 哲学叙述、6 层 audit 含 CDA、5 个 baseline 混淆变体、Publishability floor 段落**。

本工作定位明确为 **Use-Inspired 可行性研究**：testbed 是主角（痴呆护理事实冲突对话作为一个临床有据、安全张力真实的 ML 评测场景），inline annotation 是支撑该 testbed 上 RL 训练的工程通道。我们诚实声明 contribution 类型为 *integration novelty + signal-source framing + Use-Inspired empirical validation*，不主张 paradigm-level breakthrough，不主张直接临床部署，不宣称比真值更准。所有主结论以 §6.1 的 pre-registered 阈值判定，所有失败模式以 §7.1 的 fallback narrative 公开承诺。

