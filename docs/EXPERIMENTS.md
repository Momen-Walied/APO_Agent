# Experiments Plan (A/B)

- __Datasets__:
  - FiNER-ORD (gtfintechlab/finer-ord): split=train, limit=50/100/200
  - FiNER-139 (nlpaueb/finer-139): split=test/train, limit=50/100
  - AppWorld tasks via HF (instruction→response), limit=100

- __Models__:
  - Baseline single-model: 7B, 14B, 32B
  - Tiered: Gen=32B, Ref=14B, Cur=7B; + quantized variants

- __Ablations__ (toggle ONE at a time):
  1) Caches: off → traj → traj+insight → all
  2) Tiering: off → role-tiered → complexity-tiered
  3) Early stopping: off → sim≥0.95 → sim≥0.90
  4) Batching: off → batch=5 → batch=10
  5) Memory: flat → 2-tier → +distillation

- __Metrics__:
  - Task: span-F1, token accuracy; per-entity P/R/F1; bootstrap CI; McNemar p
  - Cost: tokens, $ (if cloud), calls
  - Latency: wall-time, per-stage

- __Acceptance__:
  - Cost ↓ ≥ 20% and latency ↓ ≥ 15% with accuracy loss ≤ 2% (strong); ≤ 5% (weak)

- __Plots/Tables__:
  - Pareto frontier (3D, projections)
  - Cache ablation table (hit%, Δcost, ΔF1)
  - Tiering vs accuracy/cost
  - Early-stopping ROC vs Δcost
