# Caregiver-R1 工程实施大纲

> 配套方法定义：`paper/final_rps_method_section_locked_zh_9.md`。
>
> 本文档规划工程交付路径——按"先把数据备齐，再把训练跑通"的真实顺序，从空仓库一直推到能产出 paper §6.6 "最小可发表子集"的 EC-MTRL 主结果。

## 已锁定的 architectural decisions（按时间）

下列决策在编码过程中确定，**当前是 ground truth**；paper §章节如与下面冲突，以本节 + code 为准。**PROPOSAL §9 (Decisions 1, 2, 3) 是这些决策对应到 zh_9 paper 的 supersede patch list**，本节是工程实现层视角的完整时间线。

1. **Standard GRPO/GDPO sampling**（unified prompt + 温度采样 N=10）——不走 contextual GRPO；EC-MTRL = standard GDPO + 两个扩展（Multi-horizon GDPO + α-weighted dual-horizon advantage）+ PBRS theoretical contribution + DemMA testbed contribution。**↔ PROPOSAL §9 Decision 1**.
2. **DemMA inline annotation alphabet = 34 labels**（与 hulehule/DemMA-Planner-SFT checkpoint 1:1 对齐：movement 20 + facial 7 + voice 7）。
3. **DemMA real client 已落地**（`src/data/demma_real_client.py`，transformers in-process，固定 Jacob Johnson AD-early）。
4. **(SUPERSEDED by item 5)** Single-layer safety LLM judge with 4-tier ordinal `c_safety ∈ {0,1,2,3}`, max-aggregate, top-tier (3) hard veto, tiers {1,2} via Lagrangian dual-ascent. **↔ PROPOSAL §9 Decision 2**, superseded by Decision 3.
5. **Binary hard-veto safety** (final form). `c_safety ∈ {0, 1}`，3 个 binary 灾难性 items（医疗误导 / 允许危险 / coercion-escalation, 对应 paper §4.5 P1/P2/P3），OR-aggregated。`c_safety = 1` 触发 hard veto 性能项零化 + 固定 floor 罚款 `−λ_violation` (default 5.0)。Stylistic/empathy 类问题（elderspeak / 不当 affirmation 等）保留在 R_fit 负分 criteria 中，与 c_safety 解除 double-counting。Lagrangian dual-ascent 退役（不再需要，hard-veto + 固定 floor 取代）。**↔ PROPOSAL §9 Decision 3**.

## 训练算法一句话定义

整套训练分两段：**Phase 0** 用 DemMA 配套对话做一次 SFT warm-up 得到 π₀，让 base LLM 学会稳定输出 `<think>+<response>` 两层格式（Qwen3 native thinking 让这一步成本很低）；**Phase 1** 在 π₀ 基础上跑 **standard GDPO + 两个 EC-MTRL 扩展**——算法骨架是 critic-free GRPO/GDPO（group-relative advantage，无 value head / GAE，**unified system prompt + 温度采样 N=10**），更新规则是 PPO-clip surrogate loss 加上对 π₀ 的 KL anchor，EC-MTRL 在 GDPO 之上叠两个扩展：① **Multi-horizon GDPO**（per-reward 独立归一化从 single-turn / 终端推广到 trajectory + turn 双时域）+ ② **α-weighted dual-horizon advantage**（`A = A_traj + α·A_turn`，PBRS 不变性定理保证任意 α > 0 不改最优策略集合）。zh_9 §4.7 写的 "PPO-clip" 指 surrogate loss 形式，不是经典 actor-critic PPO。**Sampling 层面与 standard GDPO 完全等价**——与 GRPO/GDPO 一样，10 条 rollout 来自 `π_θ(·|s, prompt_unified)` 同一条件分布的温度采样，prompt 内的 10 临床策略 menu 是 agent 的 *action-space prior*（agent 自由选择/混合/切换），不是 group sampling 的来源。

---

## §0 训练规模与算力预算规划（先读这一节）

本节锚定五个事实，所有规模数字都从这五个事实里推导出来：

1. **同类已发表工作的实际样本饥渴度比通用 reasoning RL 低一个量级**——**RLVER (ICLR 2026)** 用 500 scenarios + group=1 PPO 在 Qwen2.5-7B 上把 Sentient Bench 从 13.3 推到 79.2；**MAPO (zh_9 §5 最相关 baseline)** 用 **727 scenarios + group=4** 在 **Qwen3-8B/14B/32B + Qwen2.5-7B-Instruct** 全 family 上跑赢 GRPO baseline。
2. **DemMA 配套对话总量 ~3–5k dialog 是数据上限**——DemMA (ACL 2025) 的 caregiver 双侧 trace 是 SFT 唯一原生数据源（zh_9 附录 B.5），上限决定 SFT data 与 RL scenarios 的天花板。
3. **group=10 是 standard GRPO/GDPO 的 group size 选择，不是 contextual GRPO 硬约束**——临床决策空间的覆盖性需要 group 内 trajectory 之间存在足够 contrastive 行为差异，温度采样 ≥ 0.8 + group=10 是 RLVER / MAPO / GDPO 同类工作 demonstrated working 的中位数；降到 5 sample 多样性会显著下降（CRank 退化为粗 binning），但**不会**像之前 contextual GRPO framing 那样一刀打掉 method contribution。允许降 group 的场景仍仅限 PoC 档跑通流程。
4. **Qwen3 系列原生支持 hybrid `<think>` mode**（2025 年 4 月发布，今天 2026-04 已 1 年成熟）——Qwen3 base 已经原生输出 `<think>...</think>` token，与 zh_9 §4.2 R1 风格 `<think>+<response>` 设计**直接对齐**。这一事实**改变了 Phase 0 SFT 的成本结构**：原本 SFT 的首要目的是教 base 学会输出 XML 两层格式（zh_9 附录 B.5 主要诉求），在 Qwen3 上几乎免费，SFT 只剩次要目的——对齐 caregiver 临床推理风格 + 10 临床策略适应。**SFT 数据可从 ~5k 砍到 ~1–2k，epoch 砍到 1**，且把"是否需要 SFT"本身升级为 zh_9 §6.3 A12 ablation 的研究问题（无 SFT 直接 RL vs 1k minimal SFT vs 5k full SFT）。
5. **GDPO 已是 first-class supported 算法**——**NVlabs/GDPO** (NVIDIA 官方实现，431 stars，最后更新 2026-02-17) 同时提供 **verl / TRL / NeMo-RL** 三个框架的官方 example；verl 把 GDPO 放进 `AdvantageEstimator` enum，配置只需 `algorithm.adv_estimator = "gdpo"`，per-reward 独立归一化是 0 行代码原生支持。砍掉 contextual GRPO 后 EC-MTRL 的 verl patch 量 **~450 LOC**——zh_9 §4.6.1 "Multi-horizon GDPO Normalization" 的轨迹级部分（CRank for R_goal/R_fit/u_terminal）直接用 verl GDPO，我们只需在其上加：(i) 轮级 percentile rank（~150 LOC）、(ii) α-fused dual-horizon 融合（~50 LOC）、(iii) Lagrangian dual ascent for λ_safety（~150 LOC）、(iv) DemMA env adapter（~100 LOC，**已落地于 `src/data/demma_real_client.py`**，这块只剩 verl multi-turn rollout 的 hook）。Unified-prompt rollout 是 verl 标准 GRPO/GDPO 原生支持的，无需 patch。**项目起点不是 verl base config，而是 fork NVlabs/GDPO 的 verl example**，省 1–2 天起步成本。

### 三档目标定义

**能跑（PoC，~25 H100·hr，1–2 天）**——用 **Qwen3-4B**（同 Qwen3 family，原生 thinking，单 H100 跑得动，config 与 8B 100% 兼容易迁移）跑通整套 pipeline：reward / safety 规则数值范围、α-fused advantage 数值稳定性、KL anchor 是否卡死、C_cat 触发率是否合理。**不能用于发表**——group 降到 5、scenarios 降到 300、单 seed，hacking audit 大概率不过。这一档的真正用途是**Phase E.3 smoke run 之前的快速验证平台**——所有 bug 在小规模发现，再上 8B 跑正式实验。

**还可以（最小可发表，~700 H100·hr，端到端 6 周）**——对应 zh_9 §6.6 最小子集 + 校准到 RLVER/MAPO 公开规模 + Qwen3 native thinking 节省 SFT。**Qwen3-8B + 1,500 unique scenarios + group=10 + 2 epoch + 3 seed**，配 5 个核心 ablation（A1 / A3 / A8 / A9 / A12-with-no-SFT-cell）+ 3 层 hacking audit + ≥ 100 对人评。zh_9 §6.1 五个研究问题中 Q1–Q4 主线达到 pre-registered 阈值即可投 NeurIPS / ICLR main。**这是项目默认目标档位**，下面所有 Phase 的具体数字都按这一档锚定。

**还不错（Flagship，~3,500 H100·hr，端到端 ~3 个月）**——"还可以" 档主结果通过 Q1 之后再升级。zh_9 §6.2 Tier 2 baselines（MT-GRPO + RLVER）全跑、§6.3 全部 12 个 ablation 跑完、5 seed 主对比、3,000 scenarios 用满深度 testbed、外加 1–2 seed 的 **Qwen3-14B 作为 scaling figure**（DeepSeek-R1-Distill-Qwen-14B 是天然对照点；如有算力剩余可加 Qwen3-32B 作 stretch goal）。Q1–Q5 全部达阈值 + 人评 win-rate ≥ 50% vs 人类护工 reference + 临床期刊 follow-up 潜力。**不进入最初 critical path**。

### 核心规模表

| | **能跑（PoC）** | **还可以（默认目标）** | **还不错（Flagship）** |
|---|---|---|---|
| **Caregiver base model** | **Qwen3-4B** | **Qwen3-8B** | Qwen3-8B 主跑 + Qwen3-14B scaling figure（可选 32B stretch） |
| **Training judge** | Qwen3-8B | **Qwen3-32B** | Qwen3-32B + DeepSeek-V3 双判 |
| **Eval judge**（zh_9 §6.4 强制不同 family） | 同 training judge（妥协） | **Llama-3.3-70B-Instruct** | Llama-3.3-70B + GPT-4o 双判 |
| **Phase 0 — DemMA dialog 用量** | ~300 条 | **~1,500 条** | ~3,000 条 + 少量 augment |
| **Phase 0 — SFT 样本数**（Qwen3 native thinking 让 SFT 主要目的从"教 XML 格式"降为"对齐临床推理风格"） | ~500 | **~1,500–2,000** | ~5,000 |
| **Phase 0 — SFT epoch** | 1 | **1** | 1 |
| **Phase 0 — Validation Gate** | XML success rate ≥ 90% | **≥ 95%**（zh_9 §4.7 硬门槛，Qwen3 起点已接近 95%） | ≥ 98% |
| **Phase 0 — SFT GPU·hr** | ~3 | **~10** | ~30 |
| **Phase 1 — RL unique scenarios** | ~300 | **~1,500** | ~3,000（zh_9 §2.4 深度 testbed 全量） |
| **Phase 1 — Group size**（standard GRPO/GDPO group, 温度采样） | 5（妥协，仅 PoC 用） | **10** | 10 |
| **Phase 1 — Max turns / trajectory** | 8 | **10** | 10 |
| **Phase 1 — RL epoch on unique scenarios** | 1 | **2** | 3 |
| **Phase 1 — Total trajectories** | ~1.5K | **~30K** | ~90K |
| **Phase 1 — Total turns** | ~12K | **~270K** | ~810K |
| **Phase 1 — Judge call 总数**（每场景 2 次：safety + trajectory，zh_9 §4.7） | ~3K | **~60K** | ~180K |
| **Phase 1 — RL GPU·hr / seed** | ~5 | **~35** | ~90 |
| **Phase 1 — Seed 数（主跑）** | 1 | **3** | 5 |
| **Phase 1 — RL 总 GPU·hr**（含 baseline + 核心 ablation） | ~20 | **~650** | ~3,200 |
| **总算力（H100·hr）** | **~25** | **~700** | **~3,500** |
| **云价估算**（H100 ~$3/hr） | ~$75 | **~$2K** | ~$10K |
| **Wall-clock 纯训练** | 1–2 天 | **5–7 天** | ~3 周 |
| **Wall-clock 端到端**（含 DemMA 数据 + IRB + 人评） | 1 周 | **6 周** | ~3 个月 |
| **可投什么** | 内部演示 / workshop | **NeurIPS / ICLR main** | NeurIPS / ICLR main + 临床期刊 follow-up |

### 模型大小选型解释

**为什么 Qwen3 全系列而不是 Qwen2.5**。今天 2026-04，Qwen3 已发布 1 年（2025-04 release），社区事实标准已从 Qwen2.5 → Qwen3：(a) **MAPO（zh_9 §5 最相关 baseline）正经主跑就是 Qwen3-8B/14B/32B**，Qwen2.5-7B 只作 legacy 对照；(b) **Qwen3 原生 hybrid thinking mode** 直接匹配 zh_9 §4.2 R1 风格 `<think>+<response>` 设计，base 已经会输出 `<think>...</think>`，Phase 0 SFT 不用从零教格式；(c) Qwen3 在同参数下 reasoning 能力明显强于 Qwen2.5（GSM8K / MATH / MMLU-Pro 全面提升），对痴呆护理这种需要细腻临床策略 reasoning 的任务尤为重要；(d) verl / TRL / OpenRLHF 等 RL framework 已经原生支持 Qwen3 thinking mode 与 R1 风格 reward。

**为什么 PoC 档用 Qwen3-4B（而非 Qwen2.5-3B）**。同 Qwen3 family 与默认档 8B **架构 100% 兼容**，PoC 跑通的 config / prompt / reward 数值可直接 migrate 到 8B 重训，不会因为 base family 切换引入二次调优。Qwen3-4B 同样原生 thinking，单 H100 全精度跑得动 group=5 + max_turns=8，1–2 天能完成整套 reward / safety / advantage 调试。

**为什么 Qwen3-8B 是默认档**。四个 anchor：(a) MAPO 在同 base model 上做主跑，是当前 multi-turn dialog RL 的事实标准；(b) Qwen3-8B 原生 thinking 让 SFT 阶段从"教格式 + 学风格"降为"只学风格"，整体 wall-clock 节省 2–3 天；(c) 8B + group=10 + max_turns=10 在 4–8 H100 上跑 ~35 GPU·hr / seed，3 seed 主对比的预算可控；(d) Qwen3-8B 与 Qwen3-32B training judge 不同 size 但同 family，行为一致性好（judge rubric 评分风格不会因 family gap 引入 bias），同时 zh_9 §6.4 只要求 **eval judge** 与 training judge 不同 family，**training judge 与 caregiver 同 family（不同 size）完全合法**。

**为什么 Flagship 档用 Qwen3-14B 而不是 32B 作主 scaling figure**。Qwen3-14B 比 8B 算力多 ~2.2 倍（不到 4 倍——Qwen3 架构在更大尺寸下 efficiency 更好），仍能在 4–8 H100 上跑 1–2 seed，DeepSeek-R1-Distill-Qwen-14B 是天然对照点（同 base、不同 RL 路线，构成 reviewer 关心的"vs frontier-distillation"对比）；32B 算力多 ~5 倍，1 seed 也要 ~450 GPU·hr，**只在 14B 已显示明显增益且预算有剩余时再上 32B 作 stretch**。**不建议把主对比 base 换到 14B 或 32B**——5 seed 主跑预算翻 3–10 倍，性价比跌断崖。

**Judge 模型的两层 family 隔离**。caregiver Qwen3-8B → training judge Qwen3-32B（同 family 不同 size，OK）→ eval judge Llama-3.3-70B（不同 family，强制要求）。Training judge 用 Qwen3-32B 而非 Qwen2.5-32B 的理由：(a) Qwen3-32B reasoning 强 + 原生 thinking 让 rubric judge 能给出 evidence-anchored 推理（zh_9 §4.4 RULERS 风格），打分稳定性比 Qwen2.5 高；(b) Qwen3 family 内 size 差距大（8B vs 32B 训练数据 / RLHF 配方差异显著），不构成实质 self-judging bias。Judge 自部署在独立 vLLM serve（FP8 + tensor parallel），不占 caregiver 训练 GPU。

### GDPO / SFT 训练轮次的几个关键说明

**为什么 SFT 只要 ~1.5k 样本 × 1 epoch**。Qwen3 base 已经原生输出 `<think>...</think>` 两层格式，Phase 0 SFT 的目的从"**教 XML 格式**"降为"**对齐 caregiver 临床推理风格 + 10 临床策略 prompt 适应**"。MAPO 在 Qwen3 系列**完全跳过独立 SFT**直接 RL（用 Qwen3-235B 做 teacher 蒸馏后 727 样本走 GRPO），证明 Qwen3 base 不需要 cold-start SFT 就能跑 multi-turn dialog RL。我们在默认档保留 ~1.5k 样本 × 1 epoch 的 minimal SFT 是**保守稳妥**——临床推理风格与 Qwen3 base 默认风格仍有 gap（base model 没见过 NURSE / VERA / SPIKES 这些临床框架），1.5k 样本足够建立 prior。**A12 ablation 升级为研究问题**：{无 SFT 直接 RL（MAPO 风格）, 1.5k minimal SFT（默认）, 5k full SFT} 三档对比，验证 "Qwen3 native thinking 能否 obviate cold-start SFT for clinical multi-turn RL" —— 这比原 A12 "更多 SFT 是否更好" 的 trivial 对比更有 reviewer interest。

**为什么 unique scenarios 只要 1,500**。RLVER 用 500、MAPO 用 727 就在同类 benchmark 上推到 SOTA——multi-turn dialog 每条 trajectory 包含 8–10 turn × group=10 的密集 reward 信号，**有效梯度信号密度比单轮 reasoning RL 高一个量级**。1,500 unique scenarios 已经是 RLVER 的 3 倍、MAPO 的 2 倍，足够覆盖 zh_9 §2.4 深度 testbed（Medication + Identity × 3 severity，每单元 ≥ 250）。3,000 全量留给 Flagship 档做 Q5 跨亚型泛化 + Q4 安全约束的更细 ablation。

**为什么 RL epoch 只跑 2 次**。MAPO 在 Qwen3 全系列只跑 1 epoch、Qwen2.5-7B 跑 2 epoch（Qwen3 起点高所以 epoch 少）；RLVER 按 step 数训练换算下来约 1.5 epoch。multi-turn dialog RL 在同 prompt 池上跑 3+ epoch 边际收益快速衰减，且 reward hacking 风险陡升（n-gram TVD vs SFT 在 epoch 2 之后开始爆涨）。我们默认档跑 2 epoch 是 Qwen3 系列同类工作稳健中位数；若 Phase E.3 smoke run 显示 Qwen3-8B 在 1 epoch 已收敛，可考虑降到 1.5 epoch（按 step 切，节省 25% RL GPU·hr）。

**为什么 group=10 是默认而非更小**。group=10 是 **standard GRPO/GDPO 的 group size 经验中位数**——RLVER (group=1 PPO 是特例，因 simulator 自带情绪 reward 信号密度高) 之外，MAPO (group=4)、NVlabs/GDPO 官方 example (group=8)、DeepSeek-R1 zero (group=64) 横跨 4–64，10 是 multi-turn dialog 的稳健中位选择。Group 小于 5 会让 CRank/percentile rank 退化为粗 binning（5 条 trajectory 排名只有 5 档，advantage signal 噪声大；10 条有 10 档，CRank 分布接近 [-1, 1] 均匀）。Group 大于 16 在 8B 模型上不划算——effective batch / 显存 / vLLM rollout throughput 都成线性瓶颈。**允许降 group 的场景仅限 PoC 档跑通流程**（降到 5），这一档的结果不能写入 paper。

**Total rollout 数 ~30K 是什么概念**。1,500 scenarios × group=10 × 2 epoch = 30,000 trajectories，每条 trajectory ~10 turn → 300,000 turn-level 决策，对应 ~6,000 GRPO gradient step（按 effective batch=32 scenarios = 320 trajectories / step）。这个 step 数比 RLVER（~1,500 step）多 4 倍，比 MAPO（按 727 scenarios × 2 epoch / batch 算约 100–200 step）多 30 倍——主要是因为 EC-MTRL 的 dual-horizon advantage（轨迹级 + 轮级双通道独立归一化后融合）让每个 step 的 advantage 信息密度更高、需要更多 step 才能让 KL 充分扩散到 policy 各层。

**Judge call 60K 的成本控制**。zh_9 §4.7 已经把 R_goal / R_fit / u_terminal 三个分数合并到一次 judge call（vs MAPO per-turn judge 的 90 次/scenario 是 4.5× 节省），加上 safety judge 一次，每 scenario 共 2 次 judge call，1,500 × 10 × 2 epoch × 2 = 60K。按 Qwen3-32B 自部署在独立 vLLM serve（FP8 + tensor parallel + thinking mode 关闭以加速 judge），单次 judge call 约 3K 输入 token / 200 输出 token，整个 60K judge call 大约消耗 80–120 H100·hr 已经算入 "RL 总 GPU·hr" 那一行。如果用 GPT-4o-mini 商业 API 大概 ~$200，可作为 budget overrun 时的 fallback。

### 关键风险与降级路径

| 风险 | 触发信号 | 降级动作 |
|---|---|---|
| **DemMA dialog 拿到 < 1,000 条** | Phase B.1 第一周内确认 | 默认档 SFT 样本数从 1.5k 降到 500 + 启动 GPT-4o augmentation（~1 周延迟）；若仍不够则切 A12 "无 SFT 直接 RL" 档作为 primary（Qwen3 native thinking 让这成为合理 fallback 而非降级） |
| **SFT Validation Gate 不过**（XML success rate < 95%） | Phase D.2 自动验证 | Qwen3 base 起点已接近 95%，若不过通常是策略 prompt 适应问题——把 SFT 数据扩到 5k 重训（DemMA 全量），仍不过则切 A12 "无 SFT 直接 RL" 档 |
| **Smoke run 在 100 scenario 上 reward 不动** | Phase E.3 自动判定 | 优先检查 reward calibration（CRank 后分布是否符合 [-1,1] 均匀）+ KL(π||π₀) 是否被 β_KL 卡死；若 Qwen3 thinking mode 的 `<think>` 长度超出训练 max_seq_len（Qwen3 thinking 平均 1–3K token，需在 config 里给足空间） |
| **L2 simulator perturbation 跌幅 > 30%** | Phase F.4 audit | 触发 zh_9 §6.5 fallback narrative，主结论改写为 negative finding；本次训练不浪费，作为 "inline annotation 在 frozen LLM simulator 下 over-fit 的 negative result" 投 workshop |
| **Q1 win-rate 落入 [45%, 55%] 不确定区间** | Phase G.2 人评结果 | 触发 zh_9 §7.1 风险 2 fallback：claim 收缩为 "matched quality at structurally lower cost"，用 Q2/Q3/Q4 的 integration novelty 累积证据撑论文 |
| **Judge cost 超预算**（self-host vLLM 不够快） | Phase F 监控 | 切到 GPT-4o-mini API（~$200 总）作为后备，trade-off 是 self-judging bias 风险略升（GPT-4o-mini 与 caregiver Qwen3 不同 family，可接受） |
| **Qwen3 thinking mode 与训练框架不兼容** | Phase A.2 toy task 验证 | verl / TRL 当前主版本应已原生支持 Qwen3 thinking；若有 issue 可在 system prompt 强制 `/think` 开启 + 在 reward mask 中跳过 `<think>` boundary token |

### 训练基础设施架构（与 paper convention 对齐）

**起步即用 sync Ray cluster，不要做 async HTTP 优化**——这与 MAPO（verl + Qwen3-235B Judger sync）和 RLVER（verl + simulator-bundled reward）的实际工程模式一致；async judge 是工业 throughput 优化，published RL paper 几乎不做（rule-based reward 不需要、LLM-judge paper 直接付 GPU·hr 代价）。

**8×H100 单 Ray cluster 配置**（默认档）：

```
8 × H100 一个 Ray cluster:
  - 4 卡  caregiver Qwen3-8B 训练 (FSDP/DDP)
  - 2 卡  training judge Qwen3-32B (vLLM serve as Ray actor, FP8)
  - 2 卡  DemMA + safety judge (vLLM serve as Ray actor)

verl 的 reward_fn 同步调用 judge actor:
  rollout 阶段 → DemMA actor.step() sync 返回 (utterance, annotation)
  打分阶段     → judge actor.score() sync 返回 (R_goal, R_fit, u_terminal)
  整个调用走 Ray RPC，HTTP 都不用走，verl 默认就这么写
```

**为什么这个配置**：(a) verl 的 `reward_fn` 是 sync 接口，写起来直接；(b) Ray cluster 内部 RPC 比 HTTP 快、零序列化开销；(c) vLLM continuous batching 在 sync 调用下也吃满 throughput——caregiver rollout 一步出 320 trajectory，judge 一次性 score 320 条，自然 batch；(d) caregiver Qwen3-8B (16GB) + judge Qwen3-32B FP8 (32GB) + DemMA (16GB) + safety judge (复用 judge 同实例) 全装进 8×H100 (640GB) 绰绰有余。

PoC 档可以用 1–2×A100 跑 Qwen3-4B 全配置；Flagship 档 Qwen3-14B scaling figure 升到 12–16×H100。

### 一句话锚点

> **基于 NVlabs/GDPO verl example fork + EC-MTRL 三个扩展 + sync Ray cluster**：Qwen3-8B + 1,500 DemMA scenarios + group=10 + 2 epoch + 3 seed + 8×H100 单 cluster + ~700 H100·hr ≈ $2K 云算力，端到端 6 周（含 DemMA 数据 / IRB / 人评），目标 NeurIPS / ICLR main。这是当前痴呆护理 multi-turn RL 在已发表工作（MAPO + RLVER + GDPO Qwen3-family stack）规模上验证可行的最小预算。**所有下文 Phase 的具体数字都按这一档锚定**；PoC 档（Qwen3-4B）与 Flagship 档（+ Qwen3-14B scaling figure）作为预算上下限边界，分别对应 "Phase E smoke run 之前的快速验证" 和 "Phase F.1 主结果通过后的扩展实验"。

---

## Phase A — 基础设施与外部依赖

工程的起点是把"模型怎么训练"和"DemMA 怎么调用"两件事的接口先固定下来，所有后续模块都建在这两个接口之上。开发主战场是 RunPod 上的 GPU pod（不在本地 CPU 上做 dev），代码用 GitHub 私有 repo 同步。

**A.1 仓库骨架与开发环境**。`pyproject.toml` 列出最小依赖（`transformers>=4.51`、`vllm>=0.8`、`pydantic>=2.7`、`hydra-core`、`wandb`、`pytest`、`ray`），verl 从 GitHub source 安装（fork 自 NVlabs/GDPO 或 verl 主仓）。已有 `src/{data, models, rewards, strategies, training, evaluation, utils}/` 子包补齐 `__init__.py`，`tests/` 下放 schemas + DemMA mock 的 unit test。**开发环境**：本地保留一份 git 工作副本但不跑 GPU；主开发在 RunPod 1×A100 80GB pod 上（~$1.89/hr，pod 不用就 stop，persistent volume ~$7/月保 100GB 代码 + 中间数据），通过 Cursor Remote SSH 编辑；代码以 GitHub 私有 repo 为唯一真理来源。SFT 阶段切到 4×H100，主 RL 阶段切到 8×H100 单 Ray cluster。

**A.2 RL 框架选型与 NVlabs/GDPO 起点**。算法层面已锁死——critic-free GRPO + PPO-clip surrogate + KL anchor，**不带 critic / value head / GAE / entropy bonus**。框架决定就是 **verl**（理由：与 RLVER / MAPO / NVlabs/GDPO 同框架，便于 baseline 复用 + apple-to-apple 对比；Ray cluster 原生支持 sync judge actor pattern；GDPO 已 native）。**起点不是 verl base config，而是 fork `NVlabs/GDPO` 的 verl example**——它已经把 critic-free GRPO + per-reward 独立归一化（zh_9 EC-MTRL 扩展 ① 的轨迹级部分）配通，省 1–2 天起步成本。`algorithm.adv_estimator = "gdpo"` 是开关，`critic.enable = false` 必须显式关闭。在此基础上我们写 5 块 patch 代码：

| Patch | LOC | 对应 zh_9 章节 | 当前状态 |
|---|---|---|---|
| (i) 轮级 percentile rank + α-fused dual-horizon advantage 融合 | ~200 | §4.6.1 EC-MTRL 扩展 ① 轮级部分 + §4.6.2 扩展 ② | ✅ **已落地** `src/training/advantage.py` (265 LOC，含 CRank、percentile rank、α-fused、top-tier 安全零化、λ_safety 减项) |
| (ii) 把 (i) 注册为 verl `compute_advantage_ec_mtrl()` | ~50 | verl integration | 待写（GPU box ready 后做） |
| ~~(iii) Lagrangian dual ascent for λ_safety~~ | ~~~100~~ | ~~§4.5~~ | **OBSOLETE under PROPOSAL §9 Decision 3 (binary safety)** — `λ_violation` is now a fixed constant (5.0), not dual-updated; no patch needed. |
| (iv) DemMA env adapter（接入 verl multi-turn rollout loop） | ~100 | §2.3 / 附录 D.3 | DemMARealClient 已落地 (`src/data/demma_real_client.py`)；verl rollout hook 待写 |
| **合计** | **~450**（实际新增 verl patch ~250，advantage 已 done） | | |

> Unified-prompt rollout 由 verl 标准 GRPO/GDPO 原生支持（`actor_rollout_ref.rollout` 配置），不需要 patch。

完成定义：toy task（单轮 + dummy reward）上跑通 critic-free GRPO + GDPO advantage estimator，吞吐 > 100 samples/min on 1×A100，W&B 中确认无 value loss / critic loss 项；Qwen3-8B 在 verl rollout 中正确输出 `<think>+<response>` 两层（thinking mode 启用 + max_seq_len ≥ 8K）。

**A.3 DemMA Simulator 接入**。✅ **已落地（2026-04-25）**：`src/data/demma_client.py::DemMAClient` ABC + `DemMAMockClient` (CPU 随机 schema-valid annotation)；`src/data/demma_real_client.py::DemMARealClient` (transformers in-process 加载 hulehule/DemMA-Planner-SFT，包含 Qwen3-8B base + 4 层 MLP action classifier head，固定 patient_id=0 Jacob Johnson AD-early，懒加载 keep import GPU-free)；`scripts/download_demma.py` 用 `huggingface_hub.snapshot_download` 拉 ~16 GB checkpoint。**34 个 inline annotation label**（DemMA-Planner-SFT 的 `action_classifier.pt` 输出 movement 20 + facial_expression 7 + voice 7 = 34；与 zh_9 §2.3 写的 18 个不同，以 code 为准——decision 2）落到 enum + Pydantic schema (`src/data/schemas.py`)，DemMA channel name (`movement`/`facial_expression`/`voice`) ↔ schema field name (`motion`/`facial`/`sound`) 通过 `DEMMA_TO_SCHEMA_CHANNEL` 在 boundary 映射。`DemMAVLLMClient` 仍是 Phase F roadmap stub（NotImplementedError），下阶段切 vLLM serve 时再实现。每条 rollout 注入独立 `torch.manual_seed(seed)` 以避免病人侧轨迹塌缩。完成定义：✅ schema 34-label 验证通过；DemMARealClient `health_check()` + lazy load + step() 接口已就绪，等 GPU box `python scripts/download_demma.py` 后即可端到端验证。

---

## Phase B — 数据集构建

数据准备分三块：训练 SFT 用的 warm-up 数据（B.1 + B.2）、训练 RL 用的 10 临床策略 prompt 模板（B.3）、训练与评估 RL 用的场景库（B.4）。三块都必须在 Phase D 启动前就绪，否则后续训练无米下锅。

**B.1 DemMA 配套对话语料获取**。zh_9 附录 B.5 的关键工程便利在于——DemMA 论文配套的 ~3–5k dialog 已经带 caregiver 侧的 `<think>+<response>` 双侧 trace（DemMA 训练 patient simulator 时为了让 simulator 学习"在 caregiver 这么 think 时如何反应"而保留的标注），可以直接抽出来作 SFT data，零 GPT-4o 标注、零临床 RA 审核。本节的工作是把这批 dialog 落到 `data/raw/demma_dialogs/`（gitignored），并跑一份 `data_audit_report.json` 记录覆盖率、平均轮数、token 长度分布。完成定义：≥ 2,000 条 dialog 通过 schema validation。这一步是后面所有训练的 critical path——若 DemMA 数据拿不到，整个项目降级到 zh_9 附录 D.4 的第二 testbed 路径。

**B.2 SFT Warm-up 数据集构造**。`src/data/sft_extractor.py` 用 regex / XML parser 从 B.1 的 dialog 里抽出 caregiver 双侧 trace，**每条样本配同一个 unified system prompt**（`prompts/caregiver_system_prompt.md` 中 `<<<BEGIN_PROMPT>>>` … `<<<END_PROMPT>>>` 之间的内容），写成 `data/processed/sft_warmup.jsonl`，每条结构为 `{system, user, assistant}`，其中 assistant = `<think>...</think><response>...</response>`。抽取脚本用 unit test 验证 XML boundary 与 context-target 对齐。完成定义：≥ 5,000 SFT 样本，XML parse 100% 合法。

**B.3 Unified Caregiver System Prompt 锁定**。✅ **已落地（2026-04-25）**：`prompts/caregiver_system_prompt.md` 含一个英文 unified system prompt（~750 Qwen3 token），开头声明 caregiver 角色 + DemMA setting，然后列出 10 个临床策略 menu（NURSE / VERA / SPIKES / DICE / Reality Orientation / Therapeutic Fibbing / Reminiscence Therapy / Montessori / Redirection / Non-committal）作为 *agent's action-space prior*——agent 自由选择/混合/切换，prompt 不强制单一策略。同时含 strict 输出格式（`<think>` + `<response>` ≤ 2 sentences）+ 6 条 hard rules（禁止 coercion / 医学诊断 / 重复纠正 / "as an AI" 等）。loader contract: `src/training/rollout.py` 读取 `<<<BEGIN_PROMPT>>>` 与 `<<<END_PROMPT>>>` 之间的内容并 strip whitespace。**Prompt 一旦 RL run 启动就 frozen**，sha256 hash 落到 `configs/caregiver_prompt.lock.yaml` 与 run config 一同 snapshot。Group 内 10 条 trajectory 共用此 prompt，diversity 来自温度采样（≥ 0.8）+ 不同 seed。

**B.4 RL 场景库构造**。按 zh_9 §2.4 "深度 + 广度" 双层 testbed 构造结构化 scenario `q = (persona, conflict_type, severity, risk_tier)`。`src/data/scenario_builder.py` 从 DemMA persona pool 派生 + 生成 conflict 触发上下文，分别写到三个文件：`scenarios_train.jsonl`（深度 testbed：Medication + Identity × 3 severity，每单元 ≥ 500，~3,000 条，参与主对比）、`scenarios_breadth.jsonl`（广度 testbed：Temporal/Event/Spatial，每单元 ~100，~900 条，仅评估）、`scenarios_heldout.jsonl`（held-out personas，按 persona-level split 严格不重叠，用于 Q5 泛化）。完成定义：分布符合 zh_9 §2.4 表格，persona-level train/heldout 不重叠的断言通过。

---

## Phase C — Reward 与 Safety 模块（与 B 并行；第一次 RL run 之前 lock）

zh_9 §4.4 明确要求所有 reward / safety 规则在 lock 之后训练全程不修改，避免训练 / 评估循环依赖。本阶段实现 zh_9 §4.3–§4.6 的全部信号通道，每个 lock 文件都按 sha256 写到 `configs/`。

**C.1 轮级通道——Inline Annotation → Patient State Tier**（zh_9 §4.3）。`src/rewards/turn_level.py` 暴露三个核心函数：`compute_distress_tier(annotation, patient_text)` 与 `compute_resistance_tier(annotation, patient_text)` 各返回 `{0,1,2,3}` 的 ordinal tier，`compute_care_bid_mask(caregiver_response)` 返回布尔 mask（v1 rule-based 关键词 + dialogue act 抽取），三者组合出 `compute_turn_rewards(traj) -> {r_distress[t], r_resistance[t]}`，本质是 state delta 加 mask。**34 标签**到 tier 的完整 disjunctive rule 表写到 `src/rewards/clinical_anchors.yaml`，每条规则附 inline 临床引用（OERS / PAINAD / RTC anchor 文本，zh_9 附录 B.3）。单元测试覆盖每个 tier 边界与 zh_9 §4.5 P3/P4 多轮 pattern 触发情况。完成定义：rule 表 hash 写入 lock file，附录 C.1 calibration study 的 fixture 接口预留。**当前状态：未落地（P0 优先）**。

**C.2 轨迹级通道——LLM Judge**（zh_9 §4.4）。`src/rewards/trajectory_judge.py` 提供 `JudgeClient`（vLLM serve as Ray actor + 结构化 JSON 输出 + Pydantic 校验），`score_trajectory(traj)` 一次 judge call 返回 `{R_goal: 0..8, R_fit: 0..8, u_terminal: 0..4, evidence_turns: list[int]}` 三个分数（zh_9 §4.4 关键合并设计——三个 reward 共享一次 judge call 节省 API 成本）。Rubric prompt 锁定到 `src/rewards/rubrics/{r_goal,r_fit,u_terminal}.txt`（zh_9 附录 B.1），训练 judge base model + rubric hash + few-shot 样例 hash 写到 `configs/judge.lock.yaml`。重复 trajectory 经 hash 缓存避免重复打分。**训练 judge = Qwen3-32B**（FP8，单卡 H100 装得下；与 caregiver Qwen3-8B 同 family 不同 size，对照 MAPO Qwen3-235B Judger 同 family pattern）；**评估 judge = Llama-3.3-70B-Instruct**（zh_9 §6.4 强制不同 family 隔离 self-judging）。完成定义：50 条 mock trajectory 上 JSON 解析成功率 100%。

**C.3 安全 BINARY hard-veto 通道**（PROPOSAL §9 Decision 3, 2026-04-26 — supersedes Decision 2 的 4 档 ordinal 设计；最终版本）。Safety 完全从可补偿 reward 池中剥离，由**单一 trajectory-level LLM judge** 负责，但输出是 BINARY 不是 ordinal：`prompts/rubrics/c_safety.yaml` 含 **4 个 catastrophic-only criteria**，覆盖 caregiver 走向临床伤害的 3 条独立路径（endorsement / active-initiation / passive-permission，加上 coercion-escalation 的 multi-turn pattern）：(1) `catastrophic_unsafe_endorsement_or_misinformation` — 言语认可错误事实 OR 患者的危险意图；(2) `catastrophic_caregiver_initiated_unsafe_action` — caregiver 主动建议/推动患者去做危险行为；(3) `catastrophic_unsafe_action_permitted` — 被动允许患者要求的危险行为（paper §4.5 P2）；(4) `catastrophic_coercion_with_escalation` — 强制 + 下一 turn distress 升高（paper §4.5 P3 multi-turn）。OR-aggregated 到 `c_safety ∈ {0, 1}`。`c_safety = 1` 触发 hard veto——性能项乘 0 **AND** 固定 `λ_violation = 5.0` floor 罚款，确保 violator advantage = `−5` 严格低于任何 clean trajectory 的 a_traj + α·a_turn（CRank 后 bounded by ~3）。Stylistic/empathy 类问题保留在 R_fit 负分 criteria（避免与 c_safety double-counting）。base model 与 trajectory judge 严格不同 family 隔离。Rubric hash 写到 `configs/safety_judge.lock.yaml`。**这一改动让原 zh_9 §4.5 中的 `src/rewards/safety_hard.py` + `safety_predicates.py` 整组 regex 谓词以及 Lagrangian dual ascent 全部 obsolete**（已删除）；paper 重写时 §4.5 framing 从"Safe RLHF + RePO 双层"改为"binary hard-veto with fixed floor penalty"。完成定义：mock judge fixture 上 c_safety=1 触发率 ≈ 5%（3 binary criteria 各 ~2% × OR），advantage estimator 的 `c_cat_gate = (c_safety == 0)` 接口对接成功。

**C.4 GRPO 多时域归一化 + α-Weighted Advantage**（zh_9 §4.6，EC-MTRL 三个扩展中的两个）。`src/training/advantage.py` 提供 `crank(values)` 做轨迹级 Centered Rank（GOPO 风格，归一化到 `[-1,1]`）、`percentile_rank(values, mask)` 做轮级 percentile rank（支持 care-bid mask，零 mask turn 视为缺失），最后由 `compute_dual_horizon_advantage(group, alpha=1.0, lambda_safety) -> A[i,t]` 融合：

```
A[i,t] = 1[c_safety(τ_i) = 0] · (A_traj[i] + α · A_turn[i,t]) − λ_violation · c_safety(τ_i)
                ───────────────                                  ─────────────
                BINARY gate (Decision 3)                          fixed floor penalty (λ ≈ 5.0,
                                                                  NOT Lagrangian)
```

(s,t) 处有效样本 < 5 时回退为 trajectory mean baseline（zh_9 §4.6.1）。α 默认 1.0，A8 sweep 入口预留 α ∈ {0.25, 0.5, 1, 2, 4}。λ_safety 由 dual ascent 缓更新（每 K=50 步）。完成定义：合成 group=10 fixture 上手算与代码输出逐位 match。

---

## Phase D — Phase 0 SFT Warm-up

Phase D 是一次性训练 π₀——目标只有一个：让 base LLM 在 RL 启动前就稳定输出 `<think>+<response>` 两层 XML 格式，避免 RL 早期把训练步数浪费在学格式上。

`scripts/train_sft.py` 用 B.2 数据训练 π₀，`configs/sft_warmup.yaml` 锁定 base model = **Qwen3-8B**（原生 thinking mode，与 zh_9 §4.2 R1 风格 `<think>+<response>` 直接对齐——SFT 主要目的从"教 XML 格式"降为"对齐 caregiver 临床推理风格 + 10 策略适应"）、1 epoch、lr=5e-6、bs=32（zh_9 附录 B.5 规格 + Qwen3 起点高所以 epoch 减半）。训练完成后 `scripts/validate_sft.py` 在 held-out 100 prompt 上 generate → XML parse → 统计 success rate。**Validation Gate**：success rate ≥ 95%（zh_9 §4.7 / 附录 B.5 硬门槛，Qwen3 起点已接近 95%，1 epoch 通常够）；未达则增加 SFT epoch 或扩大数据量后重训，**或直接切 A12 ablation "无 SFT 直接 RL" 档**作为 fallback（Qwen3 native thinking 让这成为合理 fallback 而非降级）。达标后 π₀ checkpoint hash 写入 `configs/checkpoints.lock.yaml`，进入 Phase E。

---

## Phase E — Phase 1 RL 训练循环骨架

Phase E 把 §4.7 的 6 步循环搭起来并跑通端到端 smoke run。**关键工程纪律**：在花 5 seed × 80 GPU·hr 的算力跑主实验之前，必须先用 mock judge + 100 scenario × 1 epoch 把循环跑通，所有 bug 在小规模发现，避免主实验烧钱。

**E.1 Unified-Prompt Group Rollout**（standard GRPO/GDPO sampling）。`src/training/rollout.py` 暴露 `rollout_group(scenario, policy, demma_client, group_size=10) -> List[Trajectory]`：对每个 scenario，**同一个 unified system prompt**（`prompts/caregiver_system_prompt.md`，B.3 已锁）驱动 caregiver 与 DemMA 跑 N=10 条完整 8–10 轮对话，构成 group。10 条并行（vLLM 批处理），每条独立 random seed，温度 ≥ 0.8（diversity 来源）。每条 Trajectory 由 turn 序列组成，Turn 含 `(think, response, patient_utterance, annotation)` 四字段。完成定义：100 scenario rollout wall-clock < 30 min on 1×A100，annotation schema 100% 合法。

**E.2 GRPO 训练 Loop（基于 NVlabs/GDPO verl example fork）**（zh_9 §4.7 Phase 1）。**起点是 NVlabs/GDPO 的 verl example 配置**（critic-free GRPO + per-reward GDPO normalization 已配通）+ Phase A.2 列出的 5 块 patch（合计 ~650 LOC）。`scripts/train_grpo.py` 实现完整 6 步 loop：rollout → 安全检测（**单一 c_safety judge → 派生 c_cat_gate**）→ 性能打分（trajectory judge + 轮级 tier）→ 多时域归一化（GDPO native 做轨迹级 CRank，我们 patch 加轮级 PercRank）→ α-fused advantage → GRPO update。Loss 形式：

```
L = -E[min(r·A, clip(r, 1±ε)·A)] + β_KL · KL(π||π₀)
```

KL 锚定到 π₀，β_KL 初值 0.01（按实际 KL 动态调，目标 KL ∈ [0.01, 0.1]）。XML format token (`<think>`, `</think>`, `<response>`, `</response>`) 通过 loss mask 排除在梯度之外。Lagrangian dual ascent 每 K=50 步更新 λ_safety。Effective batch ≈ 32 scenario × 10 trajectory = 320 trajectories / step。

**Sync judge actor 模式**（与 MAPO / RLVER convention 一致）：training judge Qwen3-32B 部署为 vLLM serve Ray actor（同 cluster，2 卡），`reward_fn` 通过 `ray.get(judge_actor.score.remote(traj_batch))` sync 调用，**不写 async HTTP / retry / timeout 任何工业优化**——published RL paper 都这么做，Ray cluster 内部 RPC + vLLM continuous batching 已经吃满 throughput。判判 latency 不是瓶颈（rollout 是）。

**明确不实现**：value head / critic / GAE / entropy bonus（GRPO 设计上不需要）；async HTTP judge client（Ray actor sync 即可）。

完成定义：mock judge + 100 scenario × 5 epoch 跑通无 OOM / NaN，W&B 中确认无 value loss / critic loss / GAE 相关项。

**E.3 端到端 Smoke Run**。E.1 + E.2 + Phase C 全套 reward / safety 模块串起来，用真 judge + 真 DemMA + 100 scenario × 1 epoch 跑一次完整闭环。完成定义：1 epoch 跑完无崩溃，reward 曲线显示 learning signal（即使噪声大），**`c_safety=1` (binary catastrophic) 触发率 < 15%**（否则 LLM judge 太敏感或 RL 学坏了，回 C.3 调整 catastrophic 三项 criterion 的 anchor 文本）。Smoke run 通过即可进入 Phase F 主实验。

**E.4 W&B 监控指标**。训练侧报 loss、KL(π||π₀)、PPO clip frac、advantage mean/std (per channel)；reward 侧报六个 reward 通道分布（轨迹级 R_goal / R_fit / u_terminal + 轮级 r_distress / r_resistance + 安全 c_safety）、CRank 后分布、α-fused advantage 分布；**安全侧报 c_safety = 1 触发率（binary，目标 < 5% 训练后期），以及触发时哪条 catastrophic criterion 命中（3 项分别统计：medication / unsafe permission / coercion-escalation）**；rollout 侧报平均 turn 数、token 长度、generation throughput；同时实时跑 L1 hacking audit（n-gram TVD vs SFT），把训练中的 hacking 早期信号挂到 dashboard。`λ_violation` 是固定常数 5.0，不需要 dual-ascent 监控。

---

## Phase F — 主实验执行

Phase F 严格按 zh_9 §6.6 "最小可发表子集" 预算执行。所有训练在 E.3 smoke run 通过且全部 lock files 已写入之后启动。

**F.1 EC-MTRL 主训练**：5 seed × ~80 GPU·hr = 400 GPU·hr，输出 `checkpoints/ec_mtrl/seed_{1..5}/`。**F.2 Tier 1 Baselines**（zh_9 §6.2 最小子集）：必跑 #1 SFT only、#2 GRPO（单一终端奖励）、#4 MAPO（同代码库，匹配 GPU·hr，用于核心 Q1 inline-vs-external-judge 对照），每个 baseline 3 seed。**F.3 核心 Ablation**（zh_9 §6.3 最小子集，A9 已删除因 Decision 1；**A3 在 Decision 3 (binary safety) 下最终定义**）：A1（轮级通道开关 on/off）、**A3（安全设计：no safety / floor-only soft penalty / binary hard-veto + floor penalty，paper §4.5 Decision 3）**、A8（α sweep ∈ {0.5, 1, 2}），可选 A11（unified prompt with-menu vs without-menu，弱版本 ablation 验证 menu 的 marginal value）、A12（warm-up 充分性 {无 SFT, 1.5K, 5K}），每个 3 seed。**F.4 Reward Hacking Audit**（zh_9 §6.5）：L1 已在 E.4 实时跑，L2 用 DemMA 不同 seed / system prompt phrasing 重跑 trained policy 比较轮级 reward 跌幅，L3 抽 ≥ 50 条 "高轮级低轨迹级" rollout 人评。**触发撤回**：L2 跌幅 > 30% 或 L3 hacking-rate > 30% → 触发 zh_9 §6.5 fallback narrative，主结论改写为 negative finding。**F.5 Cross-Subtype Generalization**（Q5）：在 `scenarios_heldout.jsonl` 上跑 inference + judge 评分。

---

## Phase G — 评估

**G.1 Held-Out Judge 自动评估**。评估 judge 与训练 judge 严格不同 base model family（zh_9 §4.4 / §6.4），输出 `eval_reports/auto_eval.md`：所有方法 × 所有 metric × seed 表格。

**G.2 Pairwise Human Evaluation**（zh_9 §6.4）。≥ 100 对（最小子集，目标 ≥ 200），≥ 3 raters 含至少 1 名痴呆护理背景，rater 在 20 题 sanity 集上 Cohen's κ ≥ 0.6 才进入正式评估，按 IRB protocol 执行（zh_9 §6.4 列出招募门槛、补偿、知情同意、严重场景内容预警），输出 `eval_reports/human_eval.md`。

**G.3 Preliminary Calibration Study**（zh_9 附录 C，投稿前 commit milestone）。**C.1**：DemMA action label ↔ OERS expert annotation alignment 研究，K=500 样本，2 名临床专家独立标注，Cohen's κ 95% CI 下界须 ≥ 0.4，未达则撤回 zh_9 §1.4 属性 3 的"可被领域量表 principled 锚定"表述。**C.2（Decision 3 下最终设计）**：原 C_cat regex 谓词 audit obsolete；改为 **safety LLM judge BINARY catastrophic-vs-clean discriminator audit**——准备 ≥ 200 条 paired (catastrophic / clean) 人类标注 trajectory，验证 binary LLM judge 的 precision ≥ 0.85 / recall ≥ 0.75 / κ ≥ 0.7。未达门槛的 fallback：(i) 调整 c_safety.yaml 三个 catastrophic criterion 的 anchor 文本后重测；(ii) 若仍不达，提高 `λ_violation`（5.0 → 10.0）放大 floor penalty 容忍度，OR 添加 second-pass human-in-the-loop verification。两项 study 的结果无论达标与否都作为 paper 的 committed milestone report。

---

## 关键决策点（Phase A 末必须定稿）

| # | 决策点 | 已决 | 决策截止 |
|---|---|---|---|
| 1 | RL framework | **verl**（fork 自 NVlabs/GDPO 的 verl example） | A.2 |
| 2 | Caregiver base model | **Qwen3-8B-Instruct**（原生 thinking mode） | D.1 |
| 3 | Training judge base model | **Qwen3-32B-Instruct**（同 family 不同 size，对照 MAPO Judger pattern） | C.2 |
| 4 | Eval judge base model | **Llama-3.3-70B-Instruct**（zh_9 §6.4 强制不同 family） | C.2 |
| 5 | DemMA 部署形态 | **vLLM serve as Ray actor**（同 cluster，sync RPC 调用） | A.3 |
| 6 | 训练 cluster 配置 | **8 × H100 单 Ray cluster**（4 训练 + 2 judge + 2 DemMA/safety） | A.2 |
| 7 | Judge 调用模式 | **sync via Ray actor**（与 MAPO/RLVER 一致，不做 async HTTP） | E.2 |

定稿之前所有上下游模块按 mock 接口开发（`DemMAMockClient` 已在 `src/data/demma_client.py` 提供），定稿后一次性接入。
