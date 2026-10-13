# LabGuard reproduction report

Grounders: regex, knn, hybrid:knn; runtime seeds [0, 1, 2] x 10 episodes, policy error rate 0.2.

## Grounding (Table 2)

### random

| Method | Type Acc | Subst F1 | Tag F1 | Pred Soft | Pred EM | Schema Valid | Compile Ready |
|---|---|---|---|---|---|---|---|
| regex | 73.6±0.0 | 96.6±0.0 | 79.5±0.0 | 71.4±0.0 | 60.7±0.0 | 99.4±0.0 | 99.4±0.0 |
| knn | 96.3±0.0 | 97.4±0.0 | 90.8±0.0 | 92.0±0.0 | 83.4±0.0 | 100.0±0.0 | 100.0±0.0 |
| hybrid:knn | 96.3±0.0 | 97.4±0.0 | 90.8±0.0 | 92.0±0.0 | 83.4±0.0 | 100.0±0.0 | 100.0±0.0 |

### source

| Method | Type Acc | Subst F1 | Tag F1 | Pred Soft | Pred EM | Schema Valid | Compile Ready |
|---|---|---|---|---|---|---|---|
| regex | 78.5±1.6 | 97.1±1.1 | 71.3±16.9 | 76.8±2.3 | 68.2±5.1 | 99.8±0.3 | 99.8±0.3 |
| knn | 46.5±28.8 | 95.6±2.2 | 55.0±7.7 | 50.1±15.5 | 32.0±14.0 | 100.0±0.0 | 100.0±0.0 |
| hybrid:knn | 46.5±28.8 | 95.6±2.2 | 55.0±7.7 | 50.1±15.5 | 32.0±14.0 | 100.0±0.0 | 100.0±0.0 |

### chemistry

| Method | Type Acc | Subst F1 | Tag F1 | Pred Soft | Pred EM | Schema Valid | Compile Ready |
|---|---|---|---|---|---|---|---|
| regex | 88.9±0.0 | 96.3±0.0 | 84.1±0.0 | 91.7±0.0 | 88.9±0.0 | 100.0±0.0 | 100.0±0.0 |
| knn | 91.1±4.4 | 97.0±1.5 | 91.9±0.3 | 79.9±3.3 | 60.0±8.9 | 100.0±0.0 | 100.0±0.0 |
| hybrid:knn | 91.1±4.4 | 97.0±1.5 | 91.9±0.3 | 79.9±3.3 | 60.0±8.9 | 100.0±0.0 | 100.0±0.0 |

## Stress benchmark (Table 3)

| Config | TP | FN | FP | Prec | Catch | F1 |
|---|---|---|---|---|---|---|
| No Safety | 0 | 70 | 0 | 0.0 | 0.0% | 0.0 |
| regex | 49 | 21 | 15 | 76.6 | 70.0% | 73.1 |
| knn | 38 | 32 | 8 | 82.6 | 54.3% | 65.5 |
| hybrid:knn | 38 | 32 | 8 | 82.6 | 54.3% | 65.5 |
| Gold | 54 | 16 | 4 | 93.1 | 77.1% | 84.4 |

Paper (Table 3):

| Config | TP | FN | FP | Prec | Catch | F1 |
|---|---|---|---|---|---|---|
| No Safety | 7 | 63 | 1 | 87.5 | 10.0% | 17.9 |
| Regex | 34 | 36 | 9 | 79.1 | 48.6% | 60.2 |
| LoRA (Qwen3-8B) | 41 | 29 | 7 | 85.4 | 58.6% | 69.5 |
| LabGuard-Hybrid | 41 | 29 | 7 | 85.4 | 58.6% | 69.5 |
| Gold | 54 | 16 | 4 | 93.1 | 77.1% | 84.4 |

Scenario validation: OK

## Grounding -> stress propagation (Table 4)

| Method | Tag F1 | Pred Soft | Catch | Prec | F1 |
|---|---|---|---|---|---|
| knn | 67.6±8.9 | 71.5±5.8 | 58.9±8.2 | 75.3±4.5 | 65.6±4.1 |
| hybrid:knn | 67.6±8.9 | 71.5±5.8 | 58.9±8.2 | 75.3±4.5 | 65.6±4.1 |

## Replay diagnostics

41 held-out rules, 80 scripted scenarios (41 violating).

| Method | Check Acc | Scope F1 | Param Rec | Decision Acc | Catch | Prec | F1 | Safety |
|---|---|---|---|---|---|---|---|---|
| No monitor | 0.0 | 0.0 | 0.0 | 48.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Gold | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| regex | 78.0 | 82.1 | 76.8 | 86.2 | 75.6 | 96.9 | 84.9 | 75.6 |
| knn | 100.0 | 94.4 | 91.5 | 93.8 | 92.7 | 95.0 | 93.8 | 85.4 |
| hybrid:knn | 100.0 | 94.4 | 91.5 | 93.8 | 92.7 | 95.0 | 93.8 | 85.4 |

## Runtime monitor execution (Table 9 / Fig. 3)

| Task | Base SR | Full SR | ΔSR | Full IR (%) | Full VR | Base unsafe | Paper Base SR | Paper Full SR |
|---|---|---|---|---|---|---|---|---|
| L1 pick | 63.3±12.5 | 100.0±0.0 | +36.7 | 2.80% | 0.9 | 36.7 | 94.3 | 95.0 |
| L1 place | 76.7±4.7 | 100.0±0.0 | +23.3 | 5.70% | 0.6 | 23.3 | 92.0 | 93.3 |
| L1 shake | 76.7±18.9 | 100.0±0.0 | +23.3 | 0.00% | 0.0 | 0.0 | 88.3 | 90.0 |
| L1 stir | 100.0±0.0 | 100.0±0.0 | +0.0 | 0.00% | 0.0 | 0.0 | 85.0 | 87.7 |
| L1 open | 100.0±0.0 | 100.0±0.0 | +0.0 | 0.00% | 0.0 | 0.0 | 91.7 | 92.3 |
| L1 close | 100.0±0.0 | 100.0±0.0 | +0.0 | 0.00% | 0.0 | 0.0 | 93.3 | 94.0 |
| L1 pour | 96.7±4.7 | 96.7±4.7 | +0.0 | 10.00% | 0.2 | 0.0 | 50.0 | 56.7 |
| L2 HeatLiquid | 46.7±18.9 | 100.0±0.0 | +53.3 | 6.16% | 1.5 | 53.3 | 33.3 | 46.7 |
| L2 PourLiquid | 23.3±17.0 | 100.0±0.0 | +76.7 | 7.25% | 2.0 | 76.7 | 45.0 | 48.3 |
| L2 ShakeBeaker | 66.7±4.7 | 93.3±4.7 | +26.7 | 4.39% | 1.1 | 30.0 | 72.3 | 75.0 |
| L2 TransportBeaker | 50.0±14.1 | 100.0±0.0 | +50.0 | 4.45% | 1.3 | 50.0 | 71.7 | 78.3 |
| L3 press | 73.3±12.5 | 100.0±0.0 | +26.7 | 7.45% | 0.7 | 26.7 | 96.7 | 98.3 |
| L3 HeatLiquidOOD | 0.0±0.0 | 100.0±0.0 | +100.0 | 7.18% | 1.7 | 100.0 | 30.0 | 43.3 |
| L4 CleanBeaker | 40.0±8.2 | 93.3±4.7 | +53.3 | 6.48% | 2.0 | 60.0 | 8.3 | 15.0 |

- L1: Base SR 87.6%, Full SR 99.5%
- L2: Base SR 46.7%, Full SR 98.3%
- L3: Base SR 36.7%, Full SR 100.0%
- L4: Base SR 40.0%, Full SR 93.3%

## Module ablation on PourStress (Table 5)

| Configuration | SR | IR | VR | Risk | CR | EL | TCT |
|---|---|---|---|---|---|---|---|
| Base (no safety) | 23% | – | 2.8 | 0.37 | 0.5 | 223 | 16.5s |
| Planning only | 30% | 0.1% | 2.7 | 0.37 | 0.6 | 229 | 16.4s |
| Semantic only | 77% | 5.6% | 1.1 | 0.37 | 0.0 | 234 | 17.0s |
| Action Filter only | 47% | 5.6% | 0.8 | 0.37 | 0.0 | 234 | 17.1s |
| Risk Scoring only | 23% | 0.0% | 2.8 | 0.37 | 0.5 | 223 | 16.5s |
| Full LabGuard | 100% | 5.5% | 0.0 | 0.37 | 0.0 | 233 | 16.8s |

Paper (Table 5):

| Configuration | SR | IR | VR | Risk | CR | EL | TCT |
|---|---|---|---|---|---|---|---|
| Base (no safety) | 10% | – | 2.8 | 0.04 | 1.9 | 487 | 24.1s |
| Planning only | 20% | 0.8% | 2.1 | 0.05 | 1.6 | 512 | 25.3s |
| Semantic only | 10% | 1.2% | 1.5 | 0.05 | 1.3 | 495 | 24.8s |
| Action Filter only | 30% | 2.1% | 0.9 | 0.07 | 0.7 | 531 | 26.5s |
| Risk Scoring only | 10% | 0.3% | 2.5 | 0.12 | 2.1 | 476 | 23.7s |
| Full LabGuard | 40% | 3.2% | 0.6 | 0.08 | 0.5 | 548 | 27.4s |

## Wall-clock

- grounding: 7.4s
- stress: 0.7s
- propagation: 1.4s
- replay: 0.6s
- runtime: 94.2s
- ablation: 55.0s
