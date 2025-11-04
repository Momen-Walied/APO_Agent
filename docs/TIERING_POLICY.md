# Adaptive Model Tiering Policy

- __Role-Based__
  - Generator: Large (quality-critical)
  - Reflector: Medium (analysis/JSON)
  - Curator: Small (pattern + JSON)

- __Complexity Routing__
  - Signals: error severity, token length, #entities, prior accuracy
  - Rule: use larger reflector if severity high or repeated failure

- __Local Quantization__
  - 4-bit for reflector/curator to save VRAM, keep generator FP16
