"""Evaluation protocols of the paper (Sec. 5, Table 6).

  grounding     Table 2 / Table 6 grounding protocols (random, source-, chemistry-held-out)
  replay        replay diagnostics: 41 held-out rules x scripted scenarios (80)
  stress        100-scenario stress benchmark (Table 3) and grounding propagation (Table 4)
  runtime_eval  L1-L4 Base vs Full (Table 9, Fig. 3) and the module ablation (Table 5)
  reporting     markdown tables next to the paper's numbers
"""

from labguard.eval.grounding import (PROTOCOL_SEEDS, dev_holdout, evaluate_grounder, fit_grounder, make_builder,
                                     protocol_splits, run_protocol)
from labguard.eval.metrics import GROUNDING_KEYS, aggregate, example_scores, fmt_pm, grounding_scores, set_f1
from labguard.eval.replay import REPLAY_KEYS, ReplayRule, build_replay_set, replay_diagnostics, run_replay
from labguard.eval.runtime_eval import (ABLATION_CONFIGS, ABLATION_TASK, RUNTIME_CONFIGS, all_tasks, run_ablation,
                                        run_runtime)
from labguard.eval.scenarios import Scenario, ScenarioSampler
from labguard.eval.stress import (CATEGORIES, StressScenario, build_scenarios, run_stress, run_stress_config,
                                  stress_propagation, stress_rule_pool, validate_scenarios)

__all__ = [
    "PROTOCOL_SEEDS", "dev_holdout", "evaluate_grounder", "fit_grounder", "make_builder", "protocol_splits",
    "run_protocol", "GROUNDING_KEYS", "aggregate", "example_scores", "fmt_pm", "grounding_scores", "set_f1",
    "REPLAY_KEYS", "ReplayRule", "build_replay_set", "replay_diagnostics", "run_replay", "ABLATION_CONFIGS",
    "ABLATION_TASK", "RUNTIME_CONFIGS", "all_tasks", "run_ablation", "run_runtime", "Scenario", "ScenarioSampler",
    "CATEGORIES", "StressScenario", "build_scenarios", "run_stress", "run_stress_config", "stress_propagation",
    "stress_rule_pool", "validate_scenarios",
]
