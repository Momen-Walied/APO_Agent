# Caching Layer Blueprint

- __Trajectory Cache__
  - Key: hash( instruction + retrieved_bullets_ids + playbook_fingerprint )
  - Val: generator output JSON + confidence
  - TTL: per-dataset epoch or performance drop

- __Insight Cache__
  - Key: hash( error_signature + pred_vs_gt_delta + last_k_bullets )
  - Val: insights JSON (normalized)
  - Invalidate: when playbook fingerprint changes significantly

- __Curator Cache__
  - Key: hash( insight_summary + playbook_fingerprint )
  - Val: delta ops (ADD/UPDATE/DELETE)
  - Deterministic; long TTL with versioning

- __Staleness/Invalidation__
  - Fingerprint playbook by embedding centroids + counters
  - Drop cache when accuracy drops >5% over rolling 10 samples
