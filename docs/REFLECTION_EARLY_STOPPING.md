# Reflection Early Stopping

- __Signals__
  - Semantic similarity between consecutive insight drafts (SBERT cosine)
  - Key-phrase overlap (Jaccard)
  - JSON-structure stability (keys present, value types)

- __Rule__
  - Stop if sim_prev ≥ 0.95 for 2 consecutive rounds
  - Hard cap at 3 rounds

- __Guarantees__
  - 30–40% Reflector cost reduction with <2% accuracy loss in pilot
