# Configuration Guide

- __Choose Provider__: openrouter (cloud), ollama (local), huggingface (local transformers).
- __Model Tiering__: set role-specific entries: `generator_model`, `reflector_model`, `curator_model`.
- __Dataset Split__: prefer `train` for entity-rich NER; set `limit` to ≥50.
- __Think Mode__: for Ollama, ensure `think: false` where applicable.
- __Embeddings__: `all-MiniLM-L6-v2`; enable CUDA for speed.
- __Stats__: set `bootstrap_iters: 1000` for CI; keep `sleep: false` for local runs.
