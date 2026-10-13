# LabGuard 复现：把实验室安全规则落地为可执行的机器人运行时监控

本仓库是论文 *LabGuard*（`_ARR__SafeLab.pdf`）的代码复现。论文的流程是：

```
自然语言安全规则 ──grounder──▶ LabGuard-IR ──compiler──▶ 类型化监控器 ──runtime──▶ π_safe(s_t, a_t)
   (Sec. 3 bench)          (Sec. 4.1/4.2)            (Sec. 4.3)              (Sec. 4.4)
```

框架全部实现，可以在 CPU 上端到端运行。需要 GPU 或外部仿真器的部分（Qwen3-8B LoRA 训练、
LabUtopia/Isaac Sim 闭环）代码已写好，但本机没有实际跑过，见文末的[与论文的差异](#与论文的差异和已知限制)。

## 目录结构

```
labguard/
  ir/          LabGuard-IR：z = (t, S, p, T, v, i)，15 种约束类型、21 个任务标签、谓词 DSL、
               规范化、六行序列化、注册表（化学品/不相容表/区域/设备/PPE）            Sec. 4.1
  bench/       LabGuard-Bench 构造：203 条种子规则 → 812 条样本（模板改写 + 可选 LLM 改写）、
               模拟 5 人标注与 Fleiss κ、random / source-held-out / chemistry-held-out 划分   Sec. 3
  grounder/    规则 → IR：regex 基线、TF-IDF kNN 检索、Hybrid（神经 + 规则回退）、
               HF 生成式（零样本/少样本）、LoRA（Qwen3-8B）、语法约束解码（IRGrammar +
               LogitsProcessor）、OpenAI 兼容 API（Qwen3-32B 等）、SciBERT 编码器基线     Sec. 4.2
  compiler/    IR → 类型化监控器：15 类检查库、作用域绑定（S/T）、参数 θ 提取、
               violated() 三值判定、block/modify/log 干预与修复                     Sec. 4.3
  runtime/     SafetyShield：语义监控 + 动作过滤（工作空间/速度/加速度投影）+ 计划检查 +
               风险评分（Eq. 12：0.25ρv + 0.25ρp + 0.30ρc + 0.20ρu）+ 日志；规则 YAML 载入   Sec. 4.4
  core/        状态/动作数据结构、EvalContext（status/in_zone/distance/...）、Franka 运动学
  envs/        ToyLabEnv（轻量实验室仿真，代替 LabUtopia）、L1–L4 任务、带注入错误的脚本策略
               （ACT 替身）、episode runner 与指标、LabUtopia 适配器
  eval/        评测协议：grounding（Table 2）、压力测试（Table 3/4）、replay 诊断、
               运行时（Table 9 / Fig. 3）、模块消融（Table 5）、报告表格
configs/       运行时、LabUtopia、LoRA / LLM / SciBERT grounder 配置
scripts/       各实验入口 + run_all.py + demo.py
data/labguard_bench/   生成好的 bench（examples.jsonl、seeds.jsonl、splits.json、stats.json、annotation.json）
integrations/labutopia/  LabUtopia 入口脚本与说明
tests/         pytest（29 个测试，CPU 上约 15 秒）
results/       run_all.py 的输出（REPORT.md + report.json）
```

## 安装

```bash
pip install -r requirements.txt     # 或 pip install -e ".[neural,api,dev]"
```

核心部分（IR、编译器、运行时、bench、regex/kNN grounder、所有评测）只依赖 `numpy`、`PyYAML`、`scikit-learn`。
`torch/transformers/peft` 只有神经 grounder 需要。

> Windows 下直接从仓库根目录运行脚本即可（脚本会把仓库根目录加入 `sys.path`）；
> 若输出中文/符号乱码，设置 `PYTHONIOENCODING=utf-8`。

## 快速开始

```bash
python scripts/demo.py                  # 规则文本 → IR → 监控器 → 动作过滤 → 闭环，端到端演示
python -m pytest                        # 29 个测试
python scripts/run_all.py               # 所有 CPU 可跑的实验，写 results/REPORT.md（约 3 分钟）
python scripts/run_all.py --quick       # 冒烟测试
```

`demo.py` 第 3、4 步的实际输出（regex grounder）：

```
block   pour src=bottle_hcl dst=beaker_bleach volume=20 pour_rate=10
        -> ∅ (blocked)   fired=['R01']
modify  pour src=beaker_water dst=beaker_target volume=40 pour_rate=60
        -> pour src=beaker_water dst=beaker_target volume=40 pour_rate=25   fired=['R06']
modify  heat src=flask_ethanol dst=hot_plate_1 target_temperature=90
        -> heat src=flask_ethanol dst=hot_plate_1 target_temperature=60   fired=['R07']
modify  transport src=beaker_water dst=general_shelf speed=0.5
        -> transport src=beaker_water dst=general_shelf speed=0.25   fired=['R09']
pass    pour src=beaker_water dst=beaker_target volume=40 pour_rate=15
        -> pour src=beaker_water dst=beaker_target volume=40 pour_rate=15   fired=[]

Base      success=1/5 hazards=['collision_env', 'collision_table', 'joint_limit', 'knock_over', 'spill'] interventions=0
LabGuard  success=5/5 hazards=[] interventions=123
```

### 在代码里使用

```python
from labguard.grounder.factory import build_grounder
from labguard.runtime.shield import SafetyShield, ShieldConfig
from labguard.runtime.rules import load_rules

grounder = build_grounder("regex")                      # 或 "hybrid:knn"、"configs/grounder_lora.yaml"
ir = grounder.ground("Never heat ethanol above 60 °C.")
shield = SafetyShield.from_rules(load_rules(), grounder=grounder, config=ShieldConfig())
decision = shield.filter(state, action)                 # decision.verdict ∈ {pass, modify, block}
```

grounder spec：`regex`、`knn`、`oracle`、`hybrid:<inner>`、`lora[+cd]:<adapter 目录>`、`hf[+cd]:<模型>`、
`llm:<模型>`（OpenAI 兼容接口）、`scibert[:<checkpoint>]`，或一个 YAML/JSON 配置文件路径。`+cd` 表示语法约束解码。

## 复现各个实验

| 论文 | 命令 | 说明 |
|---|---|---|
| Table 2 grounding | `python scripts/eval_grounding.py --spec regex knn hybrid:knn --protocol all` | random（649/163）、source（4 折）、chemistry（HCl 留出，5 个种子） |
| Table 2 LoRA | `python scripts/train_grounder.py --config configs/grounder_lora.yaml --protocol random source chemistry` | 需要 CUDA；每个 split/seed 一个 adapter |
| Table 2 LLM | `python scripts/eval_grounding.py --spec configs/grounder_llm.yaml` | 需要 OpenAI 兼容服务（如 vLLM 部署 Qwen3-32B） |
| Table 3 压力测试 | `python scripts/run_stress.py` | 100 个场景（70 不安全 / 30 安全对照，7 类危险），31 条规则 |
| Table 4 传播 | `python scripts/run_stress.py --propagation --specs knn` | 用 source-held-out 训练的 grounder 重新落地规则池 |
| Replay 诊断 | `python scripts/run_replay.py` | 41 条留出规则 × 80 个脚本场景 |
| Table 9 / Fig. 3 | `python scripts/run_runtime.py` | L1–L4 共 14 个任务，Base vs Full，3 种子 × 10 episode |
| Table 5 消融 | `python scripts/run_ablation.py` | PourStress 场景，6 种模块配置 |
| 重建 bench | `python scripts/build_bench.py` | 确定性生成；`--paraphraser llm:<模型>` 用 LLM 改写 |

### LoRA（Qwen3-8B）训练

论文只说明基座是 Qwen3-8B，没有给出 LoRA 超参数。`configs/grounder_lora.yaml` 用的是常见默认值：
r=16、α=32、lr=2e-4、3 个 epoch、有效 batch 16、4-bit 量化。需要：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 版 torch，按 pytorch.org 选择与驱动匹配的版本
pip install bitsandbytes                                               # 4-bit；不装则把 load_in_4bit 设为 false
python scripts/train_grounder.py --config configs/grounder_lora.yaml --protocol random
python scripts/run_stress.py --specs regex configs/grounder_lora.yaml
```

16 GB 显存跑 Qwen3-8B 需要 4-bit 量化。训练/生成的代码路径已经用一个随机初始化的
微型 Qwen3 模型在 CPU 上测过（`tests/test_grounders.py`），但没有真正训练过 8B 模型。

### LabUtopia / Isaac Sim

见 [integrations/labutopia/README.md](integrations/labutopia/README.md)。

## 当前复现结果（`results/REPORT.md`）

CPU 上的完整结果见 `results/REPORT.md`，以下为摘要。

**Stress benchmark（Table 3）**：Gold 与论文完全一致（TP 54 / FN 16 / FP 4，F1 84.4），场景集是按论文的统计构造的。

| Config | TP | FN | FP | Catch | F1 | 论文 |
|---|---|---|---|---|---|---|
| No Safety | 0 | 70 | 0 | 0.0% | 0.0 | 7 / 63 / 1 |
| Regex | 49 | 21 | 15 | 70.0% | 73.1 | 34 / 36 / 9，F1 60.2 |
| kNN / Hybrid(kNN) | 38 | 32 | 8 | 54.3% | 65.5 | LoRA: 41 / 29 / 7，F1 69.5 |
| Gold | 54 | 16 | 4 | 77.1% | 84.4 | 54 / 16 / 4，F1 84.4 |

**Grounding（Table 2，Pred EM）**：regex 60.7（random）/ 68.2（source）/ 88.9（chemistry）；
kNN 83.4 / 32.0 / 60.0。kNN 在 random 划分上偏高：random 是按样本划分的，测试集的 127 条种子规则
全部有其他改写出现在训练集中，检索几乎等于查表。

**Replay（41 条规则 / 80 个场景）**：Gold 全部为 100；regex 决策准确率 86.2、catch 75.6；kNN 93.8 / 92.7。

**运行时（Table 9，3 种子 × 10 episode）**：各级别平均成功率 Base → Full：
L1 87.6 → 99.5，L2 46.7 → 98.3，L3 36.7 → 100，L4 40.0 → 93.3（%）。

**消融（Table 5，PourStress）**：SR Base 23% → Planning 30% → Semantic 77% → Action Filter 47% → Risk 23% → Full 100%；
Full 的 VR 为 0.0（论文 0.6）。

## 与论文的差异和已知限制

1. **Bench 是合成的。** 论文的 812 条样本来自实验室手册、PubChem GHS、专家规则和安全参考资料，并有人工标注。
   这些原始数据不可得，所以 `labguard/bench` 按论文给出的统计（203 条种子、4 个来源、15 种类型、
   每条种子 4 个改写、HCl 留出 9 条）用模板生成，标注一致性（Fleiss κ）是模拟出来的。
   所有 grounding 数字只能说明框架能跑通，不能和论文的数字直接比较。
2. **Regex 偏强。** 模板生成的文本比真实规则更规整，所以 regex 在压力测试上（F1 73.1）高于论文（60.2），
   在 chemistry 留出上也很高。
3. **没有训练 LoRA / 没有调用 LLM。** 本机 torch 是 CPU 版，也没有 bitsandbytes，所以 Table 2/3 中的 LoRA、
   Qwen3-32B 行目前空缺，用 kNN 检索作为可训练 grounder 的替身。
4. **运行时实验用 ToyLabEnv 代替 LabUtopia。** 弱策略只会因为注入的安全错误而失败，没有真实 ACT 策略的
   "能力不足"型失败，所以 Full 的成功率明显高于论文（论文中 Full 只比 Base 高 0.6–13.4 个点），
   干预率（0–10%）也高于论文（≤0.5%）。
5. **No Safety 的 TP 为 0**（论文为 7）：我们的物理模型是确定性的，不安全动作一定导致危险。
6. **消融排序不同。** 我们是 Semantic only 收益最大，论文是 Action Filter only。Risk Scoring only 与 Base 完全相同，
   因为论文附录 B.5 说风险评分"主要是诊断性的"，我们只记录、不改变动作。论文该行 IR 为 0.3%，
   说明原实现里超阈值时会触发少量安全事件，但论文没有给出触发方式，这里没有臆造。
7. **Replay 跳过 2 条退化规则**（GHS036、GHS041，形如 `not contains(src, X)` 且作用域 S={X}）：监控只在源容器含 X 时生效，
   而此时谓词必然为假，不存在合规场景，因此无法构成"违规/合规"场景对。它们被跳过，由后续规则补足到 41 条。
8. **Risk 的绝对值偏高**（各配置约 0.37，论文 0.04–0.12）。权重与 Eq. 12 一致，但论文没有给出 ρv/ρp/ρc/ρu 各自的归一化方式，
   我们的化学项直接用 GHS 危害等级，PourStress 中的危险化学品会把它推高。
9. LabUtopia 适配器只用模拟的控制器状态测过（`tests/test_labutopia_adapter.py`），没有在 Isaac Sim 里跑过。
