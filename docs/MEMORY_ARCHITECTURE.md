# Hybrid Memory Architecture

- __Short-Term (K=20)__: recent bullets, full text
- __Medium-Term (21–100)__: compressed bullets
- __Long-Term (>100)__: indexed summaries by centroid; retrieve on high similarity only

- __Pruning__
  - Remove bullets with helpful=0 and uses>10
  - Merge semantically redundant items periodically
