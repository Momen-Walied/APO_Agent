# Test Plan (Repro + Sanity)

- __Sanity__: run `verify_dataset.py` (train, 100). Ensure ≥30% sentences have entities.
- __Limit Shaping__: Confirm logs show `[Dataset] Desired samples: N, loading N*200 raw rows` and `Grouped ... into ≥N samples`.
- __Small Run__: limit=10; reflect_rounds=1; ensure playbook grows (≥1 bullet) and JSON parsing succeeds.
- __Paper-Faithful__: `configs/ollama/finer_ord_paper_faithful.yaml` with limit=50; check span-F1 improves and ≥8 bullets added.
- __Tiering Smoke__: set role-specific models; ensure per-stage usage reflects routing.
- __Label-Free__: set `label_free=true`; check BIO-consistency and agreement scores logged.
- __Reporting__: Verify report contains CI, McNemar p, per-entity metrics, and final playbook.
