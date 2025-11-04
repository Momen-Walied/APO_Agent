# Configuration Files Guide

This directory contains ready-to-use configuration files for running ACE framework experiments on different datasets and providers.

## Directory Structure

```
configs/
├── openrouter/
│   ├── finer_ord_config.yaml      # FiNER-ORD dataset with OpenRouter (DeepSeek)
│   ├── finer_139_config.yaml      # FiNER-139 dataset with OpenRouter (DeepSeek)
│   └── appworld_config.yaml       # AppWorld ReAct with OpenRouter (DeepSeek)
└── ollama/
    ├── finer_ord_config.yaml      # FiNER-ORD dataset with Ollama (Qwen2.5)
    ├── finer_139_config.yaml      # FiNER-139 dataset with Ollama (Qwen2.5)
    └── appworld_config.yaml       # AppWorld ReAct with Ollama (Qwen2.5)
```

## Prerequisites

### For OpenRouter configs:
1. Set your API key in `.env` file:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

### For Ollama configs:
1. Install Ollama: https://ollama.ai
2. Pull the model:
   ```powershell
   ollama pull qwen2.5:7b
   ```
3. Ensure Ollama daemon is running

## How to Run

### FiNER-ORD with OpenRouter
```powershell
python app.py --config configs/openrouter/finer_ord_config.yaml
```

### FiNER-139 with OpenRouter
```powershell
python app.py --config configs/openrouter/finer_139_config.yaml
```

### FiNER-ORD with Ollama (local, free)
```powershell
python app.py --config configs/ollama/finer_ord_config.yaml
```

### FiNER-139 with Ollama (local, free)
```powershell
python app.py --config configs/ollama/finer_139_config.yaml
```

### AppWorld ReAct with OpenRouter
```powershell
python app.py --config configs/openrouter/appworld_config.yaml
```

### AppWorld ReAct with Ollama
```powershell
python app.py --config configs/ollama/appworld_config.yaml
```

## Common Overrides

You can override any config parameter via CLI:

### Fast test (2 samples, no bootstrap)
```powershell
python app.py --config configs/openrouter/finer_ord_config.yaml --limit 2 --bootstrap_iters 0
```

### Label-free NER (no ground truth during adaptation)
```powershell
python app.py --config configs/openrouter/finer_ord_config.yaml --label_free true
```

### Different model
```powershell
python app.py --config configs/openrouter/finer_ord_config.yaml --model google/gemini-2.0-flash-exp:free
```

### Use cheaper models for Reflector/Curator
```powershell
python app.py --config configs/ollama/finer_ord_config.yaml --reflector_model qwen2.5:3b --curator_model qwen2.5:3b
```

### Cost tracking (OpenRouter)
```powershell
python app.py --config configs/openrouter/finer_ord_config.yaml --prompt_cost_per_1k 0.14 --completion_cost_per_1k 0.28
```

### Reproducibility with no delays
```powershell
python app.py --config configs/openrouter/finer_ord_config.yaml --seed 42 --sleep false
```

## Output Files

Each config generates a report JSON file with:
- Token-level and span-level metrics
- Per-entity precision/recall/F1
- Bootstrap confidence intervals
- Statistical significance tests (McNemar, bootstrap p-values)
- Usage statistics by stage (Generator/Reflector/Curator)
- Cost estimates (if configured)
- Final playbook content
- Metadata (seed, dataset fingerprint, providers)

## Configuration Parameters

### Core Settings
- `provider`: openrouter | ollama | gemini
- `model`: Model name for the provider
- `dataset`: HuggingFace dataset ID
- `split`: Dataset split (test | train | validation)
- `limit`: Number of samples to process

### ACE Framework
- `reflect_rounds`: Number of Reflector iterations per error (default: 3)
- `top_k`: Top-K playbook bullets to retrieve (default: 8)
- `dedup_threshold`: Embedding similarity threshold for dedup (default: 0.85)
- `embedding_model`: SentenceTransformer model (default: all-MiniLM-L6-v2)

### Evaluation
- `bootstrap_iters`: Bootstrap iterations for confidence intervals (default: 200)
- `offline_warmup_epochs`: Pre-adaptation epochs on train/validation (default: 0)
- `offline_warmup_limit`: Max warmup samples per epoch

### Label-Free NER
- `label_free`: Use label-free signals instead of ground truth (default: false)
- `bio_threshold`: BIO-consistency threshold (default: 0.9)
- `agreement_threshold`: Model agreement threshold (default: 0.8)

### AppWorld ReAct
- `run_appworld`: Run ReAct agent mode (default: false)
- `react_max_steps`: Max steps per task (default: 8)
- `appworld_hf_dataset`: HuggingFace dataset for tasks
- `appworld_hf_split`: Dataset split
- `appworld_instruction_field`: Instruction field name
- `appworld_expected_field`: Expected answer field name

### Reproducibility
- `seed`: Random seed for reproducibility
- `sleep`: Enable human-like delays (true/false)

### Cost Tracking
- `prompt_cost_per_1k`: Prompt token cost per 1k tokens
- `completion_cost_per_1k`: Completion token cost per 1k tokens
- `currency`: Currency code (e.g., USD)

## Advanced: Role-Specific Models

Save compute by using smaller models for Reflector/Curator:

```yaml
generator_provider: openrouter
generator_model: deepseek/deepseek-chat-v3.1:free
reflector_provider: ollama
reflector_model: qwen2.5:3b
curator_provider: ollama
curator_model: qwen2.5:3b
```
