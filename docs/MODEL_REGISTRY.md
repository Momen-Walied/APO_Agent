# Model Configuration Registry (Spec)

- __Schema (YAML)__:
  - `name`: human-friendly
  - `provider`: openrouter|ollama|huggingface|gemini
  - `model`: provider-specific id
  - `precision`: fp16|int8|int4
  - `quantized`: true/false
  - `role`: generator|reflector|curator|default
  - `cost`: prompt_per_1k, completion_per_1k, currency
  - `runtime`: max_new_tokens, temperature, think (provider-specific)
  - `fingerprint`: sha of (provider+model+precision+runtime)
  - `env`: CUDA version, PyTorch version

- __Naming__:
  - `tier-<S|M|L>-<provider>-<model>-<prec>`

- __Example__:
```yaml
name: tier-L-openrouter-deepseek-v3-fp16
provider: openrouter
model: deepseek/deepseek-chat-v3
precision: fp16
quantized: false
role: generator
cost:
  prompt_per_1k: 0.27
  completion_per_1k: 1.10
  currency: USD
runtime:
  max_new_tokens: 2048
  temperature: 0
  think: false
fingerprint: 1f2a...  # computed
env:
  cuda: 12.1
  pytorch: 2.4.0
```
