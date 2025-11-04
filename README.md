# MAPRO: Multi-Agent Prompt Optimization via MAP Inference (LangGraph + Ollama)

This repository implements the MAPRO framework described in `Paper_Doc/paper_doc.md` using best practices with LangGraph and Ollama. It provides:

- A pluggable Multi-Agent System (MAS) built with LangGraph
- Language-guided reward models using local Ollama models
- MAP-based prompt selection (LMPBP) and preference-guided policy updates
- A Typer CLI to initialize, optimize, and run examples

## Quickstart

1) Install Python 3.10+

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Install and run Ollama (if not installed):
- Windows: https://ollama.com/download
- Start the server: ensure `ollama serve` is running.
- Pull a model (choose one you have GPU/CPU for):

```powershell
ollama pull llama3.1:8b
# or
ollama pull qwen2.5:7b
```

4) Initialize an example workspace

```bash
python -m mapro.cli init --out workspace --model llama3.1:8b
```

5) Run optimization loop on sample tasks

```bash
python -m mapro.cli optimize \
  --config examples/config.example.yaml \
  --tasks examples/tasks/sample_math.yaml \
  --out workspace \
  --iterations 5 --patience 3 --epsilon 0.0 \
  --model llama3.1:8b
```

6) Inspect results
- Selected prompts written to `workspace/prompts.json`
- Logs and traces in `workspace/logs/`

## Project layout

- `mapro/`: core library
  - `config.py`: Pydantic configs and YAML loading
  - `ollama_client.py`: ChatOllama factory and JSON helpers
  - `policy.py`: prompt pools and persistence
  - `reward.py`: agent/edge reward scoring via LLM
  - `mutate.py`: prompt mutation utilities (trust-region)
  - `critic.py`: preference demonstration updates
  - `lmpbp.py`: MAP selection (exact on trees, beam-search for general DAG)
  - `mas.py`: example MAS (Reasoner → Extractor → Checker) with LangGraph
  - `loop.py`: the end-to-end MAPRO optimization loop
  - `cli.py`: Typer CLI entrypoints
- `examples/`: sample config and tasks
- `Paper_Doc/`: the paper text used as reference

## Notes
- This implementation is model-agnostic and uses local Ollama models via `langchain-ollama`.
- For non-tree DAGs, MAP selection uses beam search to approximate the junction-tree method described in the paper.
- The example MAS uses math QA tasks to avoid executing arbitrary code on your machine; evaluation is done by LLM-based checking and answer extraction.

## Safety
- No untrusted code execution. The example pipeline avoids running arbitrary generated code.
- If you adapt to code-generation tasks, sandbox execution appropriately.
