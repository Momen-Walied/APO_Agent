# Batching Protocols

- __Batch Reflection__
  - Input: list[ {sample_id, tokens, pred, gt, errors} ]
  - Output: list[ {sample_id, insights:[...]} ]

- __Batch Curation__
  - Input: list[ {sample_id, insights} ] + playbook_fingerprint
  - Output: { ops: [ {type, content, section, id?} ], mapping: {sample_id:[op_idx]} }

- __JSON Contracts__
  - Strict key sets; per-batch header with schema version
