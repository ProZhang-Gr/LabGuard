# 在 LabUtopia / Isaac Sim 中运行 LabGuard

论文第 5.4–5.5 节（Table 5、Table 9、Fig. 3）的闭环实验基于 [LabUtopia](https://github.com/Rui-li023/LabUtopia)（Isaac Sim 4.5 + Franka + ACT 策略）。
本目录提供把 LabGuard 插进 LabUtopia 控制回路所需的入口脚本，适配逻辑在 `labguard/envs/labutopia_adapter.py`。

> 本仓库**没有**在 Isaac Sim 中跑过这些实验（当前机器没有安装 Isaac Sim / LabUtopia），
> 适配器只在无仿真器的 mock 状态上做过单元验证。仓库内的 `ToyLabEnv` 是用来替代 LabUtopia 的
> 轻量环境，`scripts/run_runtime.py` 和 `scripts/run_ablation.py` 默认在它上面运行。

## 工作原理

LabUtopia 的 `main.py` 主循环是：

```python
state = task.step()
action, done, is_success = task_controller.step(state)
robot.get_articulation_controller().apply_action(action)
```

`main_labguard.py` 把 `main.create_controller` 替换成一个包装工厂，返回 `SafeLabUtopiaController`。每一步：

1. **状态提取**（`LabUtopiaStateExtractor`）：从 LabUtopia 的 state dict（`object_position`、`object_name`、
   `joint_positions`、`gripper_position` 等）构造 LabGuard 的 `LabState`。LabUtopia 不模拟化学过程，
   所以容器内容物来自 `object_profiles` 配置（"影子状态"），倾倒阶段结束时用语义转移模型更新。
2. **动作映射**（`LabUtopiaActionBridge`）：把 `ArticulationAction` 的关节目标通过 Franka 正运动学和
   PD 前瞻（`lookahead_steps`）转换成末端速度、目标位置等，倾倒速率按腕关节角速度 × `pour_rate_per_rad_s` 估计。
   任务标签由当前控制器阶段名映射（`PHASE_TAGS`，如 pick/pour/heat/transport）。
3. **过滤**：`SafetyShield.filter(state, action)` 依次做语义监控（block/modify）、动作过滤（工作空间、
   速度、加速度投影）和风险评分。
4. **映射回去**：被修改的动作通过逆运动学转换回关节目标；被阻止的动作变成保持当前关节位置的 hold 指令。

## 使用方法

```bash
# 1. 在 LabUtopia 的 Python 环境中安装 LabGuard
cd /path/to/labguard && pip install -e .

# 2. 把入口脚本复制（或软链接）到 LabUtopia 仓库根目录
cp integrations/labutopia/main_labguard.py /path/to/LabUtopia/

# 3. 运行（LabUtopia 自己的参数原样透传）
cd /path/to/LabUtopia
python main_labguard.py --config-name level2_PourLiquid --headless \
    --labguard-config /path/to/labguard/configs/labutopia.yaml \
    --labguard-log logs/pour_labguard.jsonl

# Base（不加安全层）对照
python main_labguard.py --config-name level2_PourLiquid --headless --labguard-off
```

LabGuard 自己的参数：

| 参数 | 含义 |
|---|---|
| `--labguard-config` | YAML 配置，包含 `shield`（各模块开关、过滤器参数）和 `adapter`（机器人基座、dt、对象 profile、场景设备） |
| `--labguard-rules` | 运行时规则 YAML，默认使用 `labguard/runtime/default_rules.yaml` |
| `--labguard-grounder` | 用 grounder 把规则文本解析成 IR（如 `regex`、`hybrid:knn`、`configs/grounder_lora.yaml`），默认直接用 gold IR |
| `--labguard-off` | 不包装控制器，即 Base 配置 |
| `--labguard-log` | 把安全日志（每一步的判定、触发规则、风险分数）写成 JSONL |

运行结束时会打印每个包装控制器的统计（步数、干预次数、阻止/修改次数、触发的规则）。

## 需要按场景调整的配置（`configs/labutopia.yaml`）

- `adapter.equipment`：场景中的设备（热板、通风橱等）的 id、类型、状态标志和世界坐标。
  设备状态类规则（如"加热前确认热板已校准"）依赖这里的信息。
- `adapter.object_profiles`：按对象名关键词指定容器种类、内容物和容量。论文中 HCl/漂白剂等
  危险组合需要在这里给对应容器设置内容物，监控才能触发。
- `adapter.zones`：工作区、储存区等区域的包围盒（基座坐标系）；不设置时使用默认工作空间。
- `shield.planning`：LabUtopia 的 ACT 控制器不暴露技能级计划，默认关闭规划检查。

## 已知限制

- 影子状态只能近似真实的化学状态；倾倒量、温度等由配置和关节运动估计，不是仿真器给出的。
- 阶段名到任务标签的映射依赖 LabUtopia 控制器的命名，新任务可能需要在 `PHASE_TAGS` 中补充。
- 论文报告的数字（Table 5/9）需要在 LabUtopia 中用 ACT checkpoint 实际运行才能复现，本仓库没有这些 checkpoint。
