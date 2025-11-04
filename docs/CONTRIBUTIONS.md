# Proposed Scientific Contributions (ACE Efficiency)

- __Resource-Efficiency Frontier__: Map the Pareto frontier of accuracy–cost–latency under model size, reflection rounds, batch size, and cache hit rate. Provide public plots and raw data.
- __Multi-Level Caching for ACE__: Design and evaluate trajectory-, insight-, and curator-caches with staleness controls and invalidation rules. Quantify hit rates and accuracy deltas.
- __Adaptive Model Tiering__: Role-aware (Generator/Reflector/Curator) and complexity-aware routing. Show that Curator can be 3–7B with <5% loss vs 30B while cutting cost 5–10×.
- __Convergence-Based Early Stopping__: Define semantic-stability metrics for reflection; stop rounds when insights converge. Theoretically motivate, empirically validate.
- __Hierarchical Memory & Distillation__: Two-tier bullets + periodic distillation. Reduce context tokens 40–70% with negligible loss.
- __Batching Protocols for Agents__: Batch reflection/curation over multiple errors to amortize system prompts and stabilize structured outputs.
- __Production Reporting__: Per-role cost/latency/tokens and statistically principled evaluation (bootstrap CI, McNemar), enabling reproducible deployments.
