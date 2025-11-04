# Reporting Templates

- __JSON Schema__ (keys already in app reports): dataset, provider(s), models, num_samples, token_acc (pre/post), span_f1 (pre/post), CI, p-values, per-entity metrics, usage.by_stage, final_playbook.

- __Markdown Summary__
```
# ACE Results – <dataset> (<provider>)
- Samples: <N>
- Span F1: <pre> → <post> (Δ=<delta>, 95% CI=<lo,hi>, p=<p>)
- Token Acc: <pre> → <post>
- Per-Entity: LOC <...>, ORG <...>, PER <...>
- Adaptation Steps Added: <k>
- Cost/Latency (by-stage): <table>
- Final Playbook (top-10):
  - [id] content
```
