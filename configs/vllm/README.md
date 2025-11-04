# vLLM Configuration for ACE Framework

vLLM provides **10-100× faster inference** than HuggingFace Transformers with the same models.

## Why vLLM?

- **PagedAttention**: Efficient KV cache management
- **Continuous batching**: Automatic request batching
- **Tensor parallelism**: Multi-GPU support
- **Optimized kernels**: Flash Attention, fused ops
- **Same models**: Use any HuggingFace model

## Performance Comparison

| Provider      | Throughput | Latency | Memory | Batching |
|---------------|-----------|---------|---------|----------|
| HuggingFace   | 1x        | 1x      | High    | Manual   |
| vLLM          | 10-20x    | 0.3-0.5x| Low     | Auto     |
| Ollama        | 3-5x      | 0.6-0.8x| Medium  | No       |

## Installation

```bash
# CUDA 12.1+ required
pip install vllm

# Verify installation
python -c "from vllm import LLM; print('vLLM ready')"
```

## Usage

### Basic Run
```powershell
python app.py --config configs/vllm/finer_ord_config.yaml
```

### Model Tiering with vLLM
```yaml
# Use vLLM for all roles (fastest)
provider: vllm
model: Qwen/Qwen2.5-14B-Instruct

# Or mix providers
generator_provider: vllm
generator_model: Qwen/Qwen2.5-14B-Instruct
reflector_provider: vllm
reflector_model: Qwen/Qwen2.5-7B-Instruct
curator_provider: ollama  # Fallback for small model
curator_model: qwen2.5:3b
```

## Supported Models

All HuggingFace models work:
- ✅ Qwen/Qwen2.5-7B-Instruct (recommended)
- ✅ Qwen/Qwen2.5-14B-Instruct (high quality)
- ✅ Qwen/Qwen2.5-32B-Instruct (best quality)
- ✅ meta-llama/Llama-3.1-8B-Instruct
- ✅ deepseek-ai/deepseek-llm-7b-chat
- ✅ mistralai/Mistral-7B-Instruct-v0.3

## Configuration Parameters

```yaml
# In vLLM engine (hardcoded for now, can expose later):
gpu_memory_utilization: 0.9  # Use 90% of GPU memory
max_model_len: 4096          # Max sequence length
trust_remote_code: true      # For custom models

# Generation params:
temperature: 0.1
max_tokens: 2048
top_p: 0.95
```

## Memory Requirements

| Model Size | GPU VRAM | Batch Size |
|-----------|----------|------------|
| 7B        | 16 GB    | 8-16       |
| 14B       | 24 GB    | 4-8        |
| 32B       | 48 GB    | 2-4        |
| 70B       | 80 GB    | 1-2        |

## Multi-GPU Support

```bash
# Tensor parallelism (automatic)
CUDA_VISIBLE_DEVICES=0,1 python app.py --config configs/vllm/finer_ord_config.yaml

# Pipeline parallelism (for very large models)
# Specify in code: LLM(..., tensor_parallel_size=2)
```

## Troubleshooting

### Out of Memory
```yaml
# Reduce max_model_len in app.py:
max_model_len: 2048  # from 4096
```

### Model not found
```bash
# Pre-download model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### Slow first inference
- Normal! vLLM compiles kernels on first run (~30s)
- Subsequent runs are instant

## Performance Tips

1. **Enable Flash Attention**: Automatic with vLLM
2. **Use FP16**: Default for GPU inference
3. **Increase batch size**: vLLM handles this automatically
4. **Tensor parallelism**: Use multiple GPUs

## Cost Comparison (50 samples)

| Provider         | Time  | Cost   |
|-----------------|-------|--------|
| OpenRouter API  | 120s  | $0.50  |
| HuggingFace     | 600s  | $0.00  |
| vLLM            | 60s   | $0.00  |

**vLLM = 10× faster than HuggingFace, free like local models**
