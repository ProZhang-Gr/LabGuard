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
