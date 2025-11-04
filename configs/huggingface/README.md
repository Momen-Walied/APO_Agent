# HuggingFace Transformers Configs

Run ACE with locally downloaded models using the HuggingFace Transformers library.

## Prerequisites

```powershell
pip install transformers torch accelerate bitsandbytes
```

For GPU support (recommended):
```powershell
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Available Models

### Recommended Models by Size:

**Small (7B) - Good for testing:**
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`

**Medium (14B) - Better quality:**
- `Qwen/Qwen2.5-14B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

**Large (32B+) - Best quality:**
- `Qwen/Qwen2.5-32B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct` (requires 40GB+ VRAM)

## Usage

### FiNER-ORD
```powershell
python app.py --config configs/huggingface/finer_ord_config.yaml
```

### FiNER-139
```powershell
python app.py --config configs/huggingface/finer_139_config.yaml
```

### AppWorld ReAct
```powershell
python app.py --config configs/huggingface/appworld_config.yaml
```

## Model Tiering for Efficiency

Use different model sizes for different roles:

```yaml
# Generator needs accuracy
provider: huggingface
model: Qwen/Qwen2.5-14B-Instruct

# Reflector can be medium
reflector_provider: huggingface
reflector_model: Qwen/Qwen2.5-7B-Instruct

# Curator can be small
curator_provider: huggingface
curator_model: meta-llama/Llama-3.2-3B-Instruct
```

## Memory Requirements

| Model Size | VRAM (FP16) | VRAM (INT8) | VRAM (INT4) |
|------------|-------------|-------------|-------------|
| 3B         | ~6 GB       | ~3 GB       | ~2 GB       |
| 7B         | ~14 GB      | ~7 GB       | ~4 GB       |
| 14B        | ~28 GB      | ~14 GB      | ~8 GB       |
| 32B        | ~64 GB      | ~32 GB      | ~16 GB      |
| 70B        | ~140 GB     | ~70 GB      | ~35 GB      |

## Quantization (Optional)

To reduce VRAM usage, use quantized models:

```yaml
# In your config, change model to:
model: TheBloke/Qwen2.5-7B-Instruct-GGUF  # 4-bit quantized
```

Or load with bitsandbytes:
```python
# Code automatically uses float16 on GPU
# For 8-bit or 4-bit, modify app.py AutoModelForCausalLM.from_pretrained:
#   load_in_8bit=True  # or load_in_4bit=True
```

## First Run

First run will download the model (~5-60 GB depending on size). Models are cached in:
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\<username>\.cache\huggingface\`

## Troubleshooting

### Out of Memory
- Use smaller model (7B instead of 14B)
- Enable quantization
- Reduce `max_new_tokens` in app.py (currently 2048)

### Slow Generation
- Ensure GPU is detected: `python check_gpu.py`
- Install CUDA-enabled PyTorch
- Use smaller model for Reflector/Curator roles

### Model Not Found
- Check model name on HuggingFace: https://huggingface.co/models
- Some models require authentication: `huggingface-cli login`
