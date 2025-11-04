#!/usr/bin/env python3
"""
evaluate_ace_gemini_finer.py

Online ACE-style evaluation using Google Gemini + HuggingFace FiNER dataset (nlpaueb/finer-139).

Usage (example):
  export GEMINI_API_KEY="your_real_key"
  python evaluate_ace_gemini_finer.py --dataset nlpaueb/finer-139 --model gemini-flash-lite-latest --limit 2
"""

import os
import json
import argparse
import time
import logging
import random
import math
import requests
import yaml
import uuid
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
from time import sleep
from dotenv import load_dotenv
from collections import Counter

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from seqeval.metrics import f1_score as seq_f1_score, precision_score as seq_precision_score, recall_score as seq_recall_score
    from seqeval.metrics.sequence_labeling import get_entities
except Exception:
    seq_f1_score = None
    seq_precision_score = None
    seq_recall_score = None
    get_entities = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

load_dotenv()

# external libs
# Note: Provider SDKs are imported lazily inside LLMClient so we can switch providers.
try:
    from datasets import load_dataset
except Exception as e:
    raise ImportError("Install datasets (pip install datasets) before running.") from e

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Sleep configuration (can be disabled)
_SLEEP_ENABLED = True
_SLEEP_MIN = 0.5
_SLEEP_MAX = 1.5


def configure_sleep(enabled: bool = True, min_s: float = 0.5, max_s: float = 1.5):
    global _SLEEP_ENABLED, _SLEEP_MIN, _SLEEP_MAX
    _SLEEP_ENABLED = bool(enabled)
    _SLEEP_MIN = float(min_s)
    _SLEEP_MAX = float(max_s)


# Small helper for human-like random delays to be gentle on APIs
def human_sleep(min_s: Optional[float] = None, max_s: Optional[float] = None):
    if not _SLEEP_ENABLED:
        return
    a = _SLEEP_MIN if min_s is None else min_s
    b = _SLEEP_MAX if max_s is None else max_s
    try:
        sleep(random.uniform(a, b))
    except Exception:
        sleep(0.2)


def parse_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


# Robust JSON parsing helpers
def parse_json_robust(text: str) -> Any:
    """Try strict parse, then fall back to raw_decode and brace scanning to extract the first JSON object/array."""
    text = (text or "").strip()
    # quick de-fence
    for token in ("```json", "```", "\u200b"):
        text = text.replace(token, "")
    text = text.strip()
    # 1) strict
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) raw_decode from start
    try:
        dec = json.JSONDecoder()
        obj, end = dec.raw_decode(text)
        return obj
    except Exception:
        pass
    # 3) find first '{' or '[' and attempt decode from there
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start != -1:
            # try raw_decode from this start
            try:
                dec = json.JSONDecoder()
                obj, end = dec.raw_decode(text[start:])
                return obj
            except Exception:
                # brace scan to find matching closer
                depth = 0
                for i in range(start, len(text)):
                    ch = text[i]
                    if ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break
    # 4) give up
    raise ValueError("Could not extract JSON from model output.")


# YAML config loader
def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config file with keys: provider, model, dataset, split, limit, output.
    Returns empty dict if path is None or file is missing/invalid."""
    cfg: Dict[str, Any] = {}
    try:
        if not path:
            # default location
            default_path = os.path.join(os.getcwd(), "config.yaml")
            if os.path.exists(default_path):
                path = default_path
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    cfg = data
                else:
                    logging.warning("YAML config is not a dict; ignoring.")
        else:
            if path:
                logging.warning(f"YAML config '{path}' not found; proceeding with CLI/env defaults.")
    except Exception as e:
        logging.warning(f"Failed to load YAML config: {e}")
    return cfg


# === 1. LLM Client (multi-provider) ===
class LLMClient:
    def __init__(self, provider: str = "gemini", model_name: str = "gemini-flash-lite-latest", api_key: Optional[str] = None, timeout: int = 300):
        self.provider = provider.lower()
        self.model_name = model_name
        self.timeout = timeout
        # usage tracking
        self.usage = {
            "calls": 0,
            "prompt_chars": 0,
            "response_chars": 0,
            "prompt_tokens_est": 0,
            "response_tokens_est": 0,
            "openrouter_prompt_tokens": 0,
            "openrouter_completion_tokens": 0,
        }
        # stage usage tracking
        self.stage_usage: Dict[str, Dict[str, Any]] = {}
        # Lazy provider-specific initialization
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
            except Exception as e:
                raise ImportError("Install google-generativeai (pip install google-generativeai) to use provider 'gemini'.") from e
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not found in environment. Set it before running.")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(model_name)
        elif self.provider == "openrouter":
            self._openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not self._openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY not found in environment. Set it before running.")
            self._openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif self.provider == "ollama":
            try:
                import ollama
            except Exception as e:
                raise ImportError("Install ollama (pip install ollama) and ensure the Ollama daemon is running to use provider 'ollama'.") from e
            self._ollama = ollama
        elif self.provider == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("HuggingFace provider requires: pip install transformers torch accelerate")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"[HuggingFace] Loading {model_name} on {device}...")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                self._hf_model = self._hf_model.to(device)
            self._hf_model.eval()
            logging.info(f"[HuggingFace] Model loaded on {device}")
        elif self.provider == "vllm":
            try:
                from vllm import LLM, SamplingParams
                self._vllm_engine = LLM
                self._vllm_sampling = SamplingParams
            except ImportError as e:
                raise RuntimeError("vLLM provider requires: pip install vllm") from e
            logging.info(f"[vLLM] Initializing {model_name}...")
            self._vllm = self._vllm_engine(
                model=model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
            )
            logging.info(f"[vLLM] Model {model_name} ready")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        logging.info(f"[LLMClient] provider={self.provider} model={model_name}")

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text or "") / 4))

    def _ensure_stage_bucket(self, stage: Optional[str]):
        key = stage or "other"
        if key not in self.stage_usage:
            self.stage_usage[key] = {
                "calls": 0,
                "prompt_chars": 0,
                "response_chars": 0,
                "prompt_tokens_est": 0,
                "response_tokens_est": 0,
                "openrouter_prompt_tokens": 0,
                "openrouter_completion_tokens": 0,
                "latency_seconds": 0.0,
            }
        return key

    def _record_usage(self, prompt: str, response_text: str, provider_usage: Optional[Dict[str, Any]] = None, stage: Optional[str] = None, latency_s: Optional[float] = None):
        self.usage["calls"] += 1
        self.usage["prompt_chars"] += len(prompt or "")
        self.usage["response_chars"] += len(response_text or "")
        self.usage["prompt_tokens_est"] += self._estimate_tokens(prompt)
        self.usage["response_tokens_est"] += self._estimate_tokens(response_text)
        if provider_usage and isinstance(provider_usage, dict):
            self.usage["openrouter_prompt_tokens"] += int(provider_usage.get("prompt_tokens", 0))
            self.usage["openrouter_completion_tokens"] += int(provider_usage.get("completion_tokens", 0))
        # stage bucket
        key = self._ensure_stage_bucket(stage)
        self.stage_usage[key]["calls"] += 1
        self.stage_usage[key]["prompt_chars"] += len(prompt or "")
        self.stage_usage[key]["response_chars"] += len(response_text or "")
        self.stage_usage[key]["prompt_tokens_est"] += self._estimate_tokens(prompt)
        self.stage_usage[key]["response_tokens_est"] += self._estimate_tokens(response_text)
        if provider_usage and isinstance(provider_usage, dict):
            self.stage_usage[key]["openrouter_prompt_tokens"] += int(provider_usage.get("prompt_tokens", 0))
            self.stage_usage[key]["openrouter_completion_tokens"] += int(provider_usage.get("completion_tokens", 0))
        if latency_s is not None:
            self.stage_usage[key]["latency_seconds"] += float(latency_s)

    def _call_model(self, prompt: str, stage: Optional[str] = None) -> str:
        if self.provider == "gemini":
            t0 = time.time()
            resp = self._gemini_model.generate_content(prompt)
            text = (resp.text or "").strip()
            self._record_usage(prompt, text, None, stage=stage, latency_s=time.time() - t0)
            return text
        elif self.provider == "openrouter":
            url = f"{self._openrouter_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._openrouter_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Reply with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
            # Use explicit (connect, read) timeouts to avoid hanging reads
            t0 = time.time()
            r = requests.post(url, headers=headers, json=payload, timeout=(10, self.timeout))
            if r.status_code != 200:
                raise RuntimeError(f"OpenRouter API error {r.status_code}: {r.text[:200]}")
            data = r.json()
            try:
                text = data["choices"][0]["message"]["content"].strip()
                self._record_usage(prompt, text, data.get("usage"), stage=stage, latency_s=time.time() - t0)
                return text
            except Exception:
                text = json.dumps(data)
                self._record_usage(prompt, text, data.get("usage") if isinstance(data, dict) else None, stage=stage, latency_s=time.time() - t0)
                return text
        elif self.provider == "ollama":
            t0 = time.time()
            # Disable thinking mode using Ollama's native think parameter
            messages = [{"role": "user", "content": prompt}]
            options = {
                "temperature": 0,
                "num_predict": 2048,
            }
            # Use think=False to disable thinking for models like deepseek-r1 and qwen3
            res = self._ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                think=False
            )
            try:
                text = res["message"]["content"].strip()
                self._record_usage(prompt, text, None, stage=stage, latency_s=time.time() - t0)
                return text
            except Exception:
                text = json.dumps(res)
                self._record_usage(prompt, text, None, stage=stage, latency_s=time.time() - t0)
                return text
        elif self.provider == "huggingface":
            t0 = time.time()
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Reply with valid JSON only when requested."},
                {"role": "user", "content": prompt}
            ]
            # Apply chat template
            input_text = self._hf_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self._hf_tokenizer(input_text, return_tensors="pt").to(self._hf_model.device)
            
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self._hf_tokenizer.eos_token_id
                )
            
            # Decode only the new tokens (exclude input)
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            text = self._hf_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            self._record_usage(prompt, text, None, stage=stage, latency_s=time.time() - t0)
            return text
        elif self.provider == "vllm":
            t0 = time.time()
            # vLLM uses SamplingParams for generation config
            sampling_params = self._vllm_sampling(
                temperature=0.1,
                max_tokens=2048,
                top_p=0.95,
            )
            # Format prompt with chat template
            formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant. Reply with valid JSON only when requested.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate (vLLM handles batching internally)
            outputs = self._vllm.generate([formatted_prompt], sampling_params)
            text = outputs[0].outputs[0].text.strip()
            
            self._record_usage(prompt, text, None, stage=stage, latency_s=time.time() - t0)
            return text
        else:
            raise ValueError("Unsupported provider")

    def generate_json(self, prompt: str, require_keys: Optional[List[str]] = None, max_retries: int = 3, sleep_between: float = 1.0, stage: Optional[str] = None) -> Dict[str, Any]:
        full_prompt = prompt + "\n\nIMPORTANT: Reply with a VALID JSON object only. No extra text before or after."
        text = ""
        for attempt in range(1, max_retries + 1):
            try:
                # brief random sleep to simulate human pacing before API call
                human_sleep(0.2, 0.8)
                text = self._call_model(full_prompt, stage=stage)
                # robust parse
                parsed = parse_json_robust(text)
                if require_keys and not all(k in parsed for k in require_keys):
                    raise ValueError(f"Missing required keys {require_keys} in model response.")
                return parsed
            except Exception as e:
                logging.warning(f"[LLMClient] attempt {attempt} failed to parse JSON: {e}")
                if attempt < max_retries:
                    # randomize retries to look more human-like
                    jitter = max(0.0, sleep_between + random.uniform(0.1, 0.6))
                    sleep(jitter)
                else:
                    logging.error("[LLMClient] All retries failed.")
                    return {"error": str(e), "raw": text}
        return {"error": "unhandled"}

    def get_usage_stats(self) -> Dict[str, Any]:
        out = dict(self.usage)
        out["by_stage"] = self.stage_usage
        return out


# === 2. Playbook ===
class ContextPlaybook:
    def __init__(self, dedup_threshold: float = 0.85, embedding_model: Optional[str] = None):
        self.bullets: List[Dict[str, Any]] = []
        self.dedup_threshold = dedup_threshold
        # Embedding backend
        self.backend = None
        self.st_model = None
        self.vectorizer = None
        self._tfidf_matrix = None
        self._emb_matrix = None  # sentence-transformers cached embeddings (n_bullets x d)
        self._init_embedding_backend(embedding_model)

    def _init_embedding_backend(self, embedding_model: Optional[str]):
        # Prefer sentence-transformers, else TF-IDF, else string similarity
        if SentenceTransformer is not None:
            try:
                # Auto-detect GPU for SentenceTransformer
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.st_model = SentenceTransformer(embedding_model or "all-MiniLM-L6-v2", device=device)
                self.backend = "st"
                logging.info(f"[Playbook] Using SentenceTransformer backend on {device.upper()}")
                return
            except Exception as e:
                logging.warning(f"SentenceTransformer init failed, falling back to TF-IDF: {e}")
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(max_features=4096)
            self.backend = "tfidf"
            logging.info("[Playbook] Using TF-IDF backend for embeddings")
        else:
            self.backend = None
            logging.warning("[Playbook] No embedding backend available; retrieval/dedup disabled")

    def _embed_all_contents(self):
        if not self.bullets:
            self._tfidf_matrix = None
            return
        contents = [b["content"] for b in self.bullets]
        if self.backend == "st":
            # compute and cache all bullet embeddings
            contents = [b["content"] for b in self.bullets]
            if contents:
                embs = self.st_model.encode(contents, normalize_embeddings=True, show_progress_bar=False)
                self._emb_matrix = np.array(embs, dtype=np.float32)
            else:
                self._emb_matrix = None
            return
        if self.backend == "tfidf":
            self._tfidf_matrix = self.vectorizer.fit_transform(contents)

    def _similarities_to_existing(self, text: str) -> List[float]:
        if not self.bullets:
            return []
        contents = [b["content"] for b in self.bullets]
        if self.backend == "st":
            try:
                if self._emb_matrix is None or len(self._emb_matrix) != len(contents):
                    self._embed_all_contents()
                q = self.st_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
                return (q @ self._emb_matrix.T).flatten().tolist()
            except Exception:
                return [SequenceMatcher(None, text, c).ratio() for c in contents]
        elif self.backend == "tfidf":
            try:
                if self._tfidf_matrix is None:
                    self._embed_all_contents()
                q = self.vectorizer.transform([text])
                sims = cosine_similarity(q, self._tfidf_matrix).flatten().tolist()
                return sims
            except Exception:
                return [SequenceMatcher(None, text, c).ratio() for c in contents]
        else:
            return [SequenceMatcher(None, text, c).ratio() for c in contents]

    def add_delta_items(self, delta_items: List[Dict[str, Any]]):
        logging.info("[Curator] Integrating delta items...")
        added = 0
        for item in delta_items:
            itype = item.get("type", "ADD").upper()
            if itype == "ADD":
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                sims = self._similarities_to_existing(content)
                if sims and max(sims) >= self.dedup_threshold:
                    logging.info("  ~ SKIP duplicate bullet (embedding-dedup)")
                    continue
                new_bullet = {
                    "id": f"bullet-{uuid.uuid4().hex[:6]}",
                    "content": content,
                    "section": item.get("section", "general"),
                    "helpful": 0,
                    "harmful": 0,
                    "uses": 0,
                }
                self.bullets.append(new_bullet)
                added += 1
                logging.info(f"  + ADDED [{new_bullet['section']}] {new_bullet['content']}")
            elif itype == "UPDATE":
                bid = item.get("id")
                for b in self.bullets:
                    if b["id"] == bid:
                        if "content" in item:
                            b["content"] = item["content"]
                        if "section" in item:
                            b["section"] = item["section"]
                        logging.info(f"  * UPDATED bullet {bid}")
                        break
            elif itype == "DELETE":
                bid = item.get("id")
                before = len(self.bullets)
                self.bullets = [b for b in self.bullets if b["id"] != bid]
                if len(self.bullets) != before:
                    logging.info(f"  - DELETED bullet {bid}")
        if added:
            self._embed_all_contents()

    def refine(self, similarity_threshold=0.92):
        # Extra string-based dedup
        if len(self.bullets) < 2:
            return
        unique = []
        seen = []
        for b in self.bullets:
            is_dup = False
            for s in seen:
                if SequenceMatcher(None, b["content"], s).ratio() > similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(b)
                seen.append(b["content"])
        if len(unique) != len(self.bullets):
            logging.info(f"[Playbook] Deduped: {len(self.bullets)} -> {len(unique)}")
            self.bullets = unique
            self._embed_all_contents()

    def get_context_string(self) -> str:
        if not self.bullets:
            return "No playbook strategies available yet."
        out = "--- CONTEXT PLAYBOOK ---\n"
        sections = sorted({b["section"] for b in self.bullets})
        for s in sections:
            out += f"\n## {s.upper()}\n"
            for b in self.bullets:
                if b["section"] == s:
                    out += f"- [{b['id']}] {b['content']}\n"
        return out

    def get_context_string_for_sample(self, sample: Dict[str, Any], top_k: int = 8, use_usefulness_weighting: bool = True) -> (str, List[str]):
        if not self.bullets:
            return self.get_context_string(), []
        query_text = " ".join(sample.get("tokens", [])) if "tokens" in sample else str(sample.get("query", ""))
        contents = [b["content"] for b in self.bullets]
        if self.backend == "st":
            if self._emb_matrix is None or len(self._emb_matrix) != len(contents):
                self._embed_all_contents()
            q = self.st_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
            sims = (q @ self._emb_matrix.T).flatten()
        elif self.backend == "tfidf":
            if self._tfidf_matrix is None:
                self._embed_all_contents()
            q = self.vectorizer.transform([query_text])
            sims = cosine_similarity(q, self._tfidf_matrix).flatten()
        else:
            sims = np.array([SequenceMatcher(None, query_text, c).ratio() for c in contents])
        
        # Usefulness weighting: combine similarity with helpful/harmful ratio
        if use_usefulness_weighting:
            usefulness_scores = np.array([
                b["helpful"] / (b["helpful"] + b["harmful"] + 1) for b in self.bullets
            ])
            # Combined score: 70% similarity + 30% usefulness
            combined_scores = 0.7 * sims + 0.3 * usefulness_scores
            idx = np.argsort(-combined_scores)[:max(1, min(top_k, len(self.bullets)))]
        else:
            idx = np.argsort(-sims)[:max(1, min(top_k, len(self.bullets)))]
        
        selected = [self.bullets[i] for i in idx]
        used_ids = [b["id"] for b in selected]
        out = "--- CONTEXT PLAYBOOK (retrieved) ---\n"
        for b in selected:
            out += f"- [{b['id']}] {b['content']}\n"
        return out, used_ids

    def update_counters(self, used_ids: List[str], helpful_delta: int = 0, harmful_delta: int = 0):
        idset = set(used_ids or [])
        for b in self.bullets:
            if b["id"] in idset:
                b["uses"] += 1
                b["helpful"] += helpful_delta
                b["harmful"] += harmful_delta


# === 3. Agentic roles ===
class Generator:
    def __init__(self, llm: LLMClient, top_k_bullets: int = 8):
        self.llm = llm
        self.top_k = top_k_bullets

    def generate(self, sample: Dict[str, Any], playbook: ContextPlaybook) -> Dict[str, Any]:
        ctx, used_ids = playbook.get_context_string_for_sample(sample, top_k=self.top_k)
        # if tokens present -> NER mode
        if "tokens" in sample and "tag_names" in sample:
            tokens = sample["tokens"]
            tag_names = sample["tag_names"]  # list of tag names ordered by index
            # prepare mapping string
            mapping_lines = "\n".join([f"{i} -> {name}" for i, name in enumerate(tag_names)])
            prompt = f"""
You are a token-level NER annotator specialized in financial entities.
Below is a mapping of integer tag indices to tag names:
{mapping_lines}

Given the list of tokens (original order), label each token with the correct integer tag index.
Use the smallest possible JSON structure.

Playbook context (if any):
{ctx}

Tokens: {tokens}

Return JSON: {{"pred_tags": [<list of integer tag indices>]}}
"""
            # Add a tiny JSON example to enforce structure
            prompt += "\nExample: {\"pred_tags\": [0, 0, 3, 4]}"
            out = self.llm.generate_json(prompt, require_keys=["pred_tags"], stage="generator")
            out["used_bullet_ids"] = used_ids
            return out
        else:
            # fallback: numeric QA/task (original behavior)
            query = sample.get("query", "")
            prompt = f"""
You are an expert financial analyst. Use the provided playbook and show your full reasoning step-by-step.

{ctx}

Task: "{query}"

Return a JSON object:
{{"reasoning": "<your step-by-step reasoning>", "final_answer": "<answer-as-string-or-number>"}}
"""
            prompt += "\nExample: {\"reasoning\": \"...\", \"final_answer\": \"42\"}"
            out = self.llm.generate_json(prompt, require_keys=["reasoning", "final_answer"], stage="generator")
            out["used_bullet_ids"] = used_ids
            return out


class Reflector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def reflect(self, sample: Dict[str, Any], trajectory: str, feedback: str, previous_insights: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        prev = previous_insights or []
        prompt = f"""
You are a Reflector agent. Analyze the sample, the generated trajectory, and the execution feedback.
Identify the root cause of the error and distill a single actionable key_insight that prevents this mistake.

Sample: {json.dumps(sample, ensure_ascii=False)}
Trajectory: "{trajectory}"
Feedback: "{feedback}"
Previous insights (for iterative refinement): {json.dumps(prev, ensure_ascii=False)}

Return JSON: {{"root_cause": "...", "key_insight": "..."}}
"""
        prompt += "\nExample: {\"root_cause\": \"...\", \"key_insight\": \"Use BIO tagging strictly ...\"}"
        return self.llm.generate_json(prompt, require_keys=["root_cause", "key_insight"], stage="reflector")


class Curator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def curate(self, insights: Dict[str, Any], playbook: ContextPlaybook) -> List[Dict[str, Any]]:
        context_str = playbook.get_context_string()
        prompt = f"""
You are a Curator agent. Given the current playbook and new insights, create incremental 'delta' updates.
- DO NOT duplicate existing content.
- Produce concise, actionable bullets.

Current Playbook:
{context_str}

Insights:
{json.dumps(insights, indent=2, ensure_ascii=False)}

Return JSON: {{"operations": [{{"type":"ADD","section":"financial_ner","content":"..."}} , ...]}}
"""
        prompt += "\nExample: {\"operations\": [{\"type\":\"ADD\",\"section\":\"financial_ner\",\"content\":\"Prefer 'B-' only at span starts; do not start 'I-'\"}]}"
        resp = self.llm.generate_json(prompt, require_keys=["operations"], stage="curator")
        return resp.get("operations", [])


# === 4. ACE framework ===
class ACEFramework:
    def __init__(self, llm_client: LLMClient, reflect_rounds: int = 1, top_k_bullets: int = 8, dedup_threshold: float = 0.85, embedding_model: Optional[str] = None,
                 generator_llm: Optional[LLMClient] = None, reflector_llm: Optional[LLMClient] = None, curator_llm: Optional[LLMClient] = None,
                 label_free: bool = False, bio_threshold: float = 0.9, agreement_threshold: float = 0.8,
                 enable_early_stopping: bool = True, convergence_threshold: float = 0.95):
        self.llm = llm_client
        self.playbook = ContextPlaybook(dedup_threshold=dedup_threshold, embedding_model=embedding_model)
        self.generator = Generator(generator_llm or llm_client, top_k_bullets=top_k_bullets)
        self.reflector = Reflector(reflector_llm or llm_client)
        self.curator = Curator(curator_llm or llm_client)
        self.reflect_rounds = max(1, int(reflect_rounds))
        self.label_free = bool(label_free)
        self.bio_threshold = float(bio_threshold)
        self.agreement_threshold = float(agreement_threshold)
        self.enable_early_stopping = bool(enable_early_stopping)
        self.convergence_threshold = float(convergence_threshold)

    def _should_curate(self, insights: Dict[str, Any]) -> bool:
        # deterministic pre-filter: minimum length and semantic novelty
        ki = (insights.get("key_insight") or "").strip()
        if len(ki) < 30:
            return False
        sims = self.playbook._similarities_to_existing(ki)
        if sims and max(sims) >= (self.playbook.dedup_threshold - 0.02):
            return False
        return True
    
    def _insight_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two insight texts using playbook's embedding backend."""
        if not text1 or not text2:
            return 0.0
        try:
            # Use playbook's embedding method if available
            if hasattr(self.playbook, '_embed_text'):
                emb1 = self.playbook._embed_text(text1)
                emb2 = self.playbook._embed_text(text2)
                if emb1 is not None and emb2 is not None:
                    # Cosine similarity
                    import numpy as np
                    norm1 = np.linalg.norm(emb1)
                    norm2 = np.linalg.norm(emb2)
                    if norm1 > 0 and norm2 > 0:
                        return float(np.dot(emb1, emb2) / (norm1 * norm2))
            # Fallback: simple string overlap
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1, text2).ratio()
        except Exception:
            return 0.0

    def run_adaptation_step(self, sample: Dict[str, Any], ground_truth: Any) -> Dict[str, Any]:
        gen_resp = self.generator.generate(sample, self.playbook)
        # NER mode
        if "tokens" in sample:
            pred_tags = gen_resp.get("pred_tags")
            final_raw = pred_tags
            parsed = None
            correct = False
            # If we have GT tags, use supervised correctness; otherwise use label-free signals
            if isinstance(pred_tags, list) and isinstance(ground_truth, list):
                parsed = pred_tags
                correct = all(int(a) == int(b) for a, b in zip(pred_tags, ground_truth))
            elif isinstance(pred_tags, list) and self.label_free:
                parsed = pred_tags
                # Label-free signals: BIO-consistency and model agreement + simple rules
                tag_names = sample.get("tag_names", [])
                pred_tagseq = intseqs_to_tagseqs([parsed], tag_names)[0]
                bio_score = bio_consistency_score(pred_tagseq)
                # agreement with alternate generator (use reflector LLM if configured)
                alt_gen = Generator(self.reflector.llm, top_k_bullets=self.generator.top_k)
                alt_resp = alt_gen.generate(sample, self.playbook)
                alt_pred = sanitize_pred_tags(alt_resp.get("pred_tags"), tokens_len=len(parsed), max_tag_idx=len(tag_names)-1 if isinstance(tag_names, list) else None)
                agree = float(sum(1 for a,b in zip(parsed, alt_pred) if int(a)==int(b)) / max(1,len(parsed)))
                # simple rules
                starts_ok = not any(p.startswith('I-') for p in pred_tagseq[:1])  # first token cannot be I-
                not_all_o = not all(t == 'O' for t in pred_tagseq)
                correct = (bio_score >= self.bio_threshold) and (agree >= self.agreement_threshold) and starts_ok and not_all_o
            self.playbook.update_counters(gen_resp.get("used_bullet_ids", []), helpful_delta=1 if correct else 0, harmful_delta=0 if correct else 1)
            if isinstance(ground_truth, list):
                feedback = "Execution SUCCEEDED." if correct else f"Execution FAILED. Expected tags: {ground_truth}, got: {pred_tags}"
            else:
                feedback = (
                    "Label-free OK." if correct else
                    "Label-free FAIL: improve BIO consistency, avoid starting with I-, and ensure predictions aren't all 'O'; also improve agreement across runs."
                )
            delta_added = False
            if not correct:
                prev_ins = []
                for round_idx in range(self.reflect_rounds):
                    insights = self.reflector.reflect(sample, str(gen_resp.get("pred_tags")), feedback, previous_insights=prev_ins)
                    
                    # Early stopping: check convergence with previous round
                    if self.enable_early_stopping and round_idx > 0 and prev_ins:
                        current_insight = insights.get("key_insight", "")
                        prev_insight = prev_ins[-1].get("key_insight", "")
                        if current_insight and prev_insight:
                            similarity = self._insight_similarity(current_insight, prev_insight)
                            if similarity >= self.convergence_threshold:
                                logging.info(f"[ACE] Early stopping at round {round_idx+1}/{self.reflect_rounds}: convergence={similarity:.3f}")
                                break
                    
                    prev_ins.append(insights)
                    if not self._should_curate(insights):
                        continue
                    ops = self.curator.curate(insights, self.playbook)
                    if ops:
                        self.playbook.add_delta_items(ops)
                        self.playbook.refine()
                        delta_added = True
                    else:
                        break
            return {
                "prediction": parsed,
                "raw": final_raw,
                "correct": correct,
                "feedback": feedback,
                "delta_added": delta_added,
                "reasoning": gen_resp.get("reasoning", "")
            }
        else:
            # numeric fallback (unchanged)
            reasoning = gen_resp.get("reasoning", "")
            final_raw = gen_resp.get("final_answer", None)
            parsed = None
            correct = False
            try:
                if isinstance(ground_truth, (int, float)):
                    parsed = float(final_raw)
                    correct = abs(parsed - float(ground_truth)) <= 1e-6
                else:
                    parsed = final_raw.strip() if isinstance(final_raw, str) else final_raw
                    correct = parsed == ground_truth
            except Exception:
                parsed = final_raw
                correct = False
            self.playbook.update_counters(gen_resp.get("used_bullet_ids", []), helpful_delta=1 if correct else 0, harmful_delta=0 if correct else 1)
            feedback = "Execution SUCCEEDED." if correct else f"Execution FAILED. Expected {ground_truth} but got {final_raw}."
            delta_added = False
            if not correct:
                prev_ins = []
                for _ in range(self.reflect_rounds):
                    insights = self.reflector.reflect({"query": sample.get("query")}, reasoning, feedback, previous_insights=prev_ins)
                    prev_ins.append(insights)
                    if not self._should_curate(insights):
                        continue
                    ops = self.curator.curate(insights, self.playbook)
                    if ops:
                        self.playbook.add_delta_items(ops)
                        self.playbook.refine()
                        delta_added = True
                    else:
                        break
            return {
                "prediction": parsed,
                "raw": final_raw,
                "correct": correct,
                "feedback": feedback,
                "delta_added": delta_added,
                "reasoning": reasoning
            }


# === 5. Dataset helpers for FiNER ===
def load_task_dataset(dataset_name: str, split: str = "test", max_rows: Optional[int] = None):
    logging.info(f"[Dataset] Loading {dataset_name} split={split} with max_rows={max_rows}...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e1:
        logging.warning(f"[Dataset] Failed with split='{split}': {e1}. Trying without split...")
        try:
            full = load_dataset(dataset_name)
            ds = load_dataset(dataset_name, split=split)
        except Exception as e2:
            logging.error(f"Failed to load dataset {dataset_name}: {e2}")
            # Try alternative dataset names
            if "finer" in dataset_name.lower():
                logging.info("Trying alternative FiNER dataset...")
                try:
                    ds = load_dataset("gtfintechlab/finer-ord", split=split)
                    logging.info("Successfully loaded gtfintechlab/finer-ord as alternative")
                except Exception as e3:
                    logging.error(f"Alternative dataset also failed: {e3}")
                    raise RuntimeError(f"Could not load any FiNER dataset. Original error: {e}")
            else:
                raise e2
    
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    logging.info(f"[Dataset] Loaded {len(ds)} samples.")
    return ds


def prepare_finer_samples(ds) -> List[Dict[str, Any]]:
    """
    Convert a FiNER example into a sample with:
      - tokens: list of tokens
      - gt_tags: list of integer tag ids
      - tag_names: full mapping list (index -> name)
    """
    samples = []
    # extract tag names from features if available
    try:
        tag_names = ds.features["ner_tags"].feature.names
    except Exception:
        try:
            # Alternative: check for gold_label feature (gtfintechlab/finer-ord format)
            tag_names = ds.features["gold_label"].feature.names
        except Exception:
            # Fallback tag names for gtfintechlab/finer-ord
            tag_names = ['O', 'PER_B', 'PER_I', 'LOC_B', 'LOC_I', 'ORG_B', 'ORG_I']

    # Group by document and sentence for gtfintechlab/finer-ord format
    if any('gold_token' in ex for ex in ds):
        # Handle gtfintechlab/finer-ord format
        doc_sentences = {}
        for ex in ds:
            doc_idx = ex.get("doc_idx", 0)
            sent_idx = ex.get("sent_idx", 0)
            token = ex.get("gold_token")
            label = ex.get("gold_label")
            
            if token is None or label is None:
                continue
                
            key = (doc_idx, sent_idx)
            if key not in doc_sentences:
                doc_sentences[key] = {"tokens": [], "labels": []}
            
            doc_sentences[key]["tokens"].append(token)
            doc_sentences[key]["labels"].append(label)
        
        # Convert grouped sentences to samples
        for (doc_idx, sent_idx), data in doc_sentences.items():
            if data["tokens"] and data["labels"]:
                sample = {
                    "tokens": data["tokens"],
                    "gt_tags": data["labels"],
                    "tag_names": tag_names
                }
                samples.append(sample)
    else:
        # Handle original nlpaueb/finer-139 format
        for ex in ds:
            tokens = ex.get("tokens")
            ner_tags = ex.get("ner_tags")
            if not tokens or ner_tags is None:
                continue
            sample = {
                "tokens": tokens,
                "gt_tags": ner_tags,
                "tag_names": tag_names
            }
            samples.append(sample)
    
    return samples


# === Metrics helpers ===
def compute_token_micro_f1(preds_list: List[Any], gold_list: List[List[int]], o_index: int = 0) -> float:
    """
    Compute token-level micro F1 excluding the 'O' label (index o_index).
    preds_list: list of per-sample predicted tag lists (may contain None);
    gold_list: list of per-sample ground truth tag lists.
    """
    tp = fp = fn = 0
    for preds, gold in zip(preds_list, gold_list):
        if not isinstance(preds, list) or not isinstance(gold, list):
            continue
        L = min(len(preds), len(gold))
        for i in range(L):
            try:
                p = int(preds[i])
                g = int(gold[i])
            except Exception:
                continue
            if g != o_index and p == g:
                tp += 1
            if p != o_index and p != g:
                fp += 1
            if g != o_index and p != g:
                fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sanitize_pred_tags(preds: Any, tokens_len: int, max_tag_idx: Optional[int] = None, default_idx: int = 0) -> List[int]:
    """Cast to ints, clamp to range, and align length to tokens_len by truncation/padding."""
    out: List[int] = []
    if not isinstance(preds, list):
        return [default_idx] * tokens_len
    for x in preds:
        try:
            v = int(x)
        except Exception:
            v = default_idx
        if max_tag_idx is not None:
            if v < 0:
                v = default_idx
            elif v > max_tag_idx:
                v = default_idx
        out.append(v)
    # align length
    if len(out) > tokens_len:
        out = out[:tokens_len]
    elif len(out) < tokens_len:
        out.extend([default_idx] * (tokens_len - len(out)))
    return out


# === Span-level metrics and significance ===
def _to_bio(tag_name: str) -> str:
    if not isinstance(tag_name, str) or not tag_name:
        return 'O'
    t = tag_name.strip()
    if t == 'O':
        return 'O'
    if '-' in t and (t.startswith('B-') or t.startswith('I-')):
        return t
    if t.endswith('_B'):
        return 'B-' + t[:-2]
    if t.endswith('_I'):
        return 'I-' + t[:-2]
    # fallback: treat as B-<TYPE>
    return 'B-' + t


def intseqs_to_tagseqs(int_seqs: List[List[int]], tag_names: List[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for seq in int_seqs:
        cur = []
        for v in seq:
            try:
                idx = int(v)
                if 0 <= idx < len(tag_names):
                    cur.append(_to_bio(tag_names[idx]))
                else:
                    cur.append('O')
            except Exception:
                cur.append('O')
        out.append(cur)
    return out


def _extract_spans_bio(tags: List[str]) -> List[tuple]:
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith('B-'):
            t = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f'I-{t}':
                j += 1
            spans.append((t, i, j - 1))
            i = j
        else:
            i += 1
    return spans


def compute_span_micro_f1_from_tags(pred_tag_seqs: List[List[str]], gold_tag_seqs: List[List[str]]) -> float:
    if seq_f1_score is not None:
        try:
            return float(seq_f1_score(gold_tag_seqs, pred_tag_seqs))
        except Exception:
            pass
    # Fallback exact span matching
    tp = fp = fn = 0
    for pseq, gseq in zip(pred_tag_seqs, gold_tag_seqs):
        p_spans = set(_extract_spans_bio(pseq))
        g_spans = set(_extract_spans_bio(gseq))
        tp += len(p_spans & g_spans)
        fp += len(p_spans - g_spans)
        fn += len(g_spans - p_spans)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_span_per_entity_metrics_from_tags(pred_tag_seqs: List[List[str]], gold_tag_seqs: List[List[str]]) -> Dict[str, Dict[str, float]]:
    if get_entities is not None:
        try:
            per_type = {}
            for preds, golds in zip(pred_tag_seqs, gold_tag_seqs):
                p_spans = set(get_entities(preds))  # (type, start, end)
                g_spans = set(get_entities(golds))
                types = set([t for t, _, _ in p_spans] + [t for t, _, _ in g_spans])
                for t in types:
                    tp = len([1 for s in p_spans if s in g_spans and s[0] == t])
                    fp = len([1 for s in p_spans if s[0] == t and s not in g_spans])
                    fn = len([1 for s in g_spans if s[0] == t and s not in p_spans])
                    if t not in per_type:
                        per_type[t] = Counter()
                    per_type[t]['tp'] += tp
                    per_type[t]['fp'] += fp
                    per_type[t]['fn'] += fn
            out = {}
            for t, c in per_type.items():
                tp, fp, fn = c.get('tp', 0), c.get('fp', 0), c.get('fn', 0)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                out[t] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
            return out
        except Exception:
            pass
    # Fallback
    per_type = {}
    for preds, golds in zip(pred_tag_seqs, gold_tag_seqs):
        p_spans = set(_extract_spans_bio(preds))
        g_spans = set(_extract_spans_bio(golds))
        types = set([t for t, _, _ in p_spans] + [t for t, _, _ in g_spans])
        for t in types:
            tp = len([1 for s in p_spans if s in g_spans and s[0] == t])
            fp = len([1 for s in p_spans if s[0] == t and s not in g_spans])
            fn = len([1 for s in g_spans if s[0] == t and s not in p_spans])
            if t not in per_type:
                per_type[t] = Counter()
            per_type[t]['tp'] += tp
            per_type[t]['fp'] += fp
            per_type[t]['fn'] += fn
    out = {}
    for t, c in per_type.items():
        tp, fp, fn = c.get('tp', 0), c.get('fp', 0), c.get('fn', 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[t] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return out


def bootstrap_ci(metric_fn, preds, golds, iters: int = 300, alpha: float = 0.05, rng_seed: int = 42) -> Dict[str, float]:
    if iters <= 0 or len(preds) == 0:
        return {"low": None, "high": None}
    rng = np.random.default_rng(rng_seed)
    n = len(preds)
    vals = np.zeros(iters, dtype=float)
    idxs = np.arange(n)
    for k in range(iters):
        S = rng.choice(idxs, size=n, replace=True)
        pv = [preds[i] for i in S]
        gv = [golds[i] for i in S]
        vals[k] = float(metric_fn(pv, gv))
    lo = float(np.percentile(vals, 100 * (alpha / 2)))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return {"low": lo, "high": hi}


def bootstrap_pvalue_improvement(metric_fn, preds_a, preds_b, golds, iters: int = 300, rng_seed: int = 43) -> float:
    if iters <= 0 or len(preds_a) == 0:
        return None
    rng = random.Random(rng_seed)
    n = len(preds_a)
    idxs = list(range(n))
    win = 0
    for _ in range(iters):
        S = [rng.choice(idxs) for _ in range(n)]
        pa = [preds_a[i] for i in S]
        pb = [preds_b[i] for i in S]
        g = [golds[i] for i in S]
        da = metric_fn(pa, g)
        db = metric_fn(pb, g)
        if (db - da) <= 0:
            win += 1
    return win / iters


def mcnemar_pvalue(b: int, c: int) -> float:
    # Use chi-square with continuity correction approximation; 1 DOF
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # p ≈ erfc(sqrt(chi2/2)) for chi-square 1 df
    try:
        from math import erfc, sqrt
        return erfc((chi2 / 2) ** 0.5)
    except Exception:
        # fallback crude
        return math.exp(-chi2 / 2)


# Label-free signals helpers
def bio_consistency_score(tag_seq_bio: List[str]) -> float:
    if not tag_seq_bio:
        return 0.0
    invalid = 0
    for i, tag in enumerate(tag_seq_bio):
        if tag.startswith('I-'):
            t = tag[2:]
            prev = tag_seq_bio[i-1] if i>0 else 'O'
            if not (prev == f'B-{t}' or prev == f'I-{t}'):
                invalid += 1
    return max(0.0, 1.0 - invalid / max(1,len(tag_seq_bio)))


# === AppWorld ReAct (lightweight integration) ===
class DummyAppWorldEnv:
    def __init__(self):
        self.state = {}
    def reset(self, instruction: str, metadata: Optional[Dict[str,Any]]=None):
        self.state = {"instruction": instruction, "history": []}
        return self.state
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # Echo tool call; in a real env, perform tool/effect
        self.state["history"].append(action)
        return {"observation": f"ok:{action.get('tool')}"}
    def success(self, final_answer: str, expected: Optional[str]) -> bool:
        if expected is None:
            return bool(final_answer and len(final_answer) > 0)
        return expected.lower() in (final_answer or '').lower()


class SimpleReActAgent:
    def __init__(self, llm: LLMClient, max_steps: int = 8):
        self.llm = llm
        self.max_steps = max_steps
    def run(self, instruction: str, env: DummyAppWorldEnv, expected: Optional[str] = None) -> Dict[str, Any]:
        env.reset(instruction)
        steps = []
        final = None
        for t in range(self.max_steps):
            prompt = f"""
You are a ReAct agent. Think step-by-step, then choose an action or give a final answer.
Instruction: {instruction}
Return JSON: {{"thought":"...", "action": {{"tool":"<name>", "input":"..."}}}} OR {{"final":"..."}}
"""
            out = self.llm.generate_json(prompt, require_keys=None, stage="agent")
            if isinstance(out, dict) and "final" in out:
                final = out.get("final")
                steps.append({"final": final})
                break
            act = (out or {}).get("action", {})
            if not isinstance(act, dict) or "tool" not in act:
                # default to no-op
                final = (out or {}).get("thought") or ""
                steps.append({"final": final})
                break
            obs = env.step(act)
            steps.append({"thought": out.get("thought"), "action": act, "observation": obs})
        success = env.success(final, expected)
        return {"steps": steps, "final": final, "success": success}


def run_appworld_react(tasks: List[Dict[str, Any]], agent_llm: LLMClient, max_steps: int = 8) -> Dict[str, Any]:
    env = DummyAppWorldEnv()
    agent = SimpleReActAgent(agent_llm, max_steps=max_steps)
    results = []
    succ = 0
    for task in tasks:
        instruction = task.get("instruction") or task.get("query") or ""
        expected = task.get("expected")
        r = agent.run(instruction, env, expected=expected)
        results.append({"instruction": instruction, **r})
        succ += 1 if r.get("success") else 0
    tgc = succ / max(1,len(tasks))
    return {"TGC": tgc, "results": results}


def load_appworld_tasks_from_hf(dataset_id: str, split: str = "test", instruction_field: Optional[str] = None,
                                expected_field: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load tasks from a HuggingFace dataset. We expect at least an instruction-like field, and optionally an expected field.
    Heuristics try common field names when not provided.
    """
    ds = load_dataset(dataset_id, split=split)
    tasks = []
    instr_keys = [instruction_field] if instruction_field else [
        "instruction", "query", "goal", "prompt", "task", "input"
    ]
    exp_keys = [expected_field] if expected_field else [
        "expected", "answer", "final", "target", "label"
    ]
    n = len(ds)
    maxn = min(n, limit) if limit else n
    for i, ex in enumerate(ds):
        if i >= maxn:
            break
        inst = None
        exp = None
        for k in instr_keys:
            if k and k in ex and ex[k]:
                inst = ex[k]
                break
        for k in exp_keys:
            if k and k in ex and ex[k] is not None:
                exp = ex[k]
                break
        if inst is None:
            continue
        tasks.append({"instruction": inst, "expected": exp})
    return tasks


# === Cost estimation ===
def estimate_cost(usage: Dict[str, Any], prompt_rate_per_1k: Optional[float], completion_rate_per_1k: Optional[float]) -> Optional[Dict[str, Any]]:
    if prompt_rate_per_1k is None or completion_rate_per_1k is None:
        return None
    # Prefer exact OpenRouter token counts if available; otherwise use estimated tokens
    prompt_toks = usage.get("openrouter_prompt_tokens") or usage.get("prompt_tokens_est") or 0
    completion_toks = usage.get("openrouter_completion_tokens") or usage.get("response_tokens_est") or 0
    cost_prompt = (prompt_toks / 1000.0) * float(prompt_rate_per_1k)
    cost_completion = (completion_toks / 1000.0) * float(completion_rate_per_1k)
    total = cost_prompt + cost_completion
    return {
        "prompt_tokens": int(prompt_toks),
        "completion_tokens": int(completion_toks),
        "prompt_cost": round(cost_prompt, 6),
        "completion_cost": round(cost_completion, 6),
        "total_cost": round(total, 6),
    }


def aggregate_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "calls": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "prompt_tokens_est": 0,
        "response_tokens_est": 0,
        "openrouter_prompt_tokens": 0,
        "openrouter_completion_tokens": 0,
        "by_stage": {}
    }
    for u in usages:
        if not isinstance(u, dict):
            continue
        out["calls"] += int(u.get("calls", 0))
        out["prompt_chars"] += int(u.get("prompt_chars", 0))
        out["response_chars"] += int(u.get("response_chars", 0))
        out["prompt_tokens_est"] += int(u.get("prompt_tokens_est", 0))
        out["response_tokens_est"] += int(u.get("response_tokens_est", 0))
        out["openrouter_prompt_tokens"] += int(u.get("openrouter_prompt_tokens", 0))
        out["openrouter_completion_tokens"] += int(u.get("openrouter_completion_tokens", 0))
        per = u.get("by_stage", {})
        if isinstance(per, dict):
            for k, v in per.items():
                b = out["by_stage"].setdefault(k, {
                    "calls": 0,
                    "prompt_chars": 0,
                    "response_chars": 0,
                    "prompt_tokens_est": 0,
                    "response_tokens_est": 0,
                    "openrouter_prompt_tokens": 0,
                    "openrouter_completion_tokens": 0,
                    "latency_seconds": 0.0,
                })
                b["calls"] += int(v.get("calls", 0))
                b["prompt_chars"] += int(v.get("prompt_chars", 0))
                b["response_chars"] += int(v.get("response_chars", 0))
                b["prompt_tokens_est"] += int(v.get("prompt_tokens_est", 0))
                b["response_tokens_est"] += int(v.get("response_tokens_est", 0))
                b["openrouter_prompt_tokens"] += int(v.get("openrouter_prompt_tokens", 0))
                b["openrouter_completion_tokens"] += int(v.get("openrouter_completion_tokens", 0))
                b["latency_seconds"] += float(v.get("latency_seconds", 0.0))
    return out


def estimate_cost_by_stage(usage_by_stage: Dict[str, Any], prompt_rate_per_1k: Optional[float], completion_rate_per_1k: Optional[float]) -> Optional[Dict[str, Any]]:
    if prompt_rate_per_1k is None or completion_rate_per_1k is None:
        return None
    out = {}
    for stage, v in (usage_by_stage or {}).items():
        prompt_toks = v.get("openrouter_prompt_tokens") or v.get("prompt_tokens_est") or 0
        completion_toks = v.get("openrouter_completion_tokens") or v.get("response_tokens_est") or 0
        cost_prompt = (prompt_toks / 1000.0) * float(prompt_rate_per_1k)
        cost_completion = (completion_toks / 1000.0) * float(completion_rate_per_1k)
        total = cost_prompt + cost_completion
        out[stage] = {
            "prompt_tokens": int(prompt_toks),
            "completion_tokens": int(completion_toks),
            "prompt_cost": round(cost_prompt, 6),
            "completion_cost": round(cost_completion, 6),
            "total_cost": round(total, 6),
            "latency_seconds": round(float(v.get("latency_seconds", 0.0)), 3),
        }
    return out


# === 6. Evaluation loop (online) for FiNER ===
def evaluate_online_with_gemini(dataset_name: str, provider: str, model_name: str, split: str = "test", limit: Optional[int] = None,
                                reflect_rounds: int = 1, top_k: int = 8, dedup_threshold: float = 0.85, embedding_model: Optional[str] = None,
                                offline_warmup_epochs: int = 0, offline_warmup_limit: Optional[int] = None,
                                bootstrap_iters: int = 0,
                                prompt_cost_per_1k: Optional[float] = None,
                                completion_cost_per_1k: Optional[float] = None,
                                currency: str = "USD",
                                seed: Optional[int] = None,
                                sleep_enabled: Optional[bool] = True,
                                generator_provider: Optional[str] = None, generator_model: Optional[str] = None,
                                reflector_provider: Optional[str] = None, reflector_model: Optional[str] = None,
                                curator_provider: Optional[str] = None, curator_model: Optional[str] = None,
                                label_free: bool = False, bio_threshold: float = 0.9, agreement_threshold: float = 0.8):
    # seeds and sleep
    if seed is not None:
        try:
            random.seed(seed)
            np.random.seed(seed)
        except Exception:
            pass
    configure_sleep(enabled=parse_bool(sleep_enabled, True))

    # LLMs (role-specific optional)
    llm = LLMClient(provider=provider, model_name=model_name)
    g_llm = LLMClient(provider=generator_provider or provider, model_name=generator_model or model_name) if (generator_provider or generator_model) else llm
    r_llm = LLMClient(provider=reflector_provider or provider, model_name=reflector_model or model_name) if (reflector_provider or reflector_model) else llm
    c_llm = LLMClient(provider=curator_provider or provider, model_name=curator_model or model_name) if (curator_provider or curator_model) else llm
    # Phase 1 optimizations
    enable_early_stopping = cfg.get("enable_early_stopping", True)
    convergence_threshold = float(cfg.get("convergence_threshold", 0.95))
    
    ace = ACEFramework(llm, reflect_rounds=reflect_rounds, top_k_bullets=top_k, dedup_threshold=dedup_threshold, embedding_model=embedding_model,
                       generator_llm=g_llm, reflector_llm=r_llm, curator_llm=c_llm,
                       label_free=label_free, bio_threshold=bio_threshold, agreement_threshold=agreement_threshold,
                       enable_early_stopping=enable_early_stopping, convergence_threshold=convergence_threshold)

    # For finer-ord format, each row is one token, so we need to load more rows to get desired sentence count
    # Heuristic: load limit*200 rows (conservative estimate to ensure enough complete sentences)
    raw_limit = limit * 200 if limit else None
    logging.info(f"[Dataset] Desired samples: {limit}, loading {raw_limit} raw rows for token-per-row datasets")
    ds = load_task_dataset(dataset_name, split=split, max_rows=raw_limit)
    logging.info(f"[Dataset] Full dataset has {len(ds)} rows")
    
    # If limit specified, take subset of raw dataset before grouping
    if raw_limit and len(ds) > raw_limit:
        ds = ds.select(range(min(raw_limit, len(ds))))
        logging.info(f"[Dataset] Selected first {len(ds)} rows for grouping")
    
    # detect if dataset is FiNER-like (tokens + ner_tags)
    # if so, prepare accordingly
    samples = []
    # try FiNER
    samples = prepare_finer_samples(ds)
    logging.info(f"[Dataset] Grouped {len(ds)} rows into {len(samples)} samples (sentences)")
    if not samples:
        # fallback: try original numeric extractor
        samples = []
        for ex in ds:
            q = ex.get("question") or ex.get("query") or ex.get("input") or None
            gt = ex.get("answer") or ex.get("label") or None
            if q is None or gt is None:
                continue
            try:
                if isinstance(gt, str) and gt.replace(".", "", 1).replace("-", "", 1).isdigit():
                    gt = float(gt)
            except:
                pass
            samples.append({"query": q, "gt": gt})

    # Apply sample-level limit (after grouping/conversion)
    if limit:
        samples = samples[:min(limit, len(samples))]

    if not samples:
        raise RuntimeError("No suitable samples extracted. Adapt field mapping for this dataset.")
    logging.info(f"[Eval] Using {len(samples)} samples for evaluation.")

    # Optional offline warmup (train or validation)
    offline_warmup_info = {"epochs": 0, "steps_added": 0, "time_seconds": 0.0}
    if offline_warmup_epochs and offline_warmup_epochs > 0:
        ds_warm = None
        for warm_split in ("train", "validation"):
            try:
                ds_warm = load_task_dataset(dataset_name, split=warm_split)
                break
            except Exception:
                continue
        if ds_warm is not None:
            warm_samples = prepare_finer_samples(ds_warm)
            if offline_warmup_limit:
                warm_samples = warm_samples[:min(offline_warmup_limit, len(warm_samples))]
            t_w0 = time.time()
            steps = 0
            for ep in range(offline_warmup_epochs):
                random.shuffle(warm_samples)
                for s in warm_samples:
                    res = ace.run_adaptation_step(s, s.get("gt_tags", s.get("gt")))
                    if res.get("delta_added"):
                        steps += 1
                    human_sleep(0.2, 0.7)
            offline_warmup_info = {"epochs": offline_warmup_epochs, "steps_added": steps, "time_seconds": round(time.time() - t_w0, 3)}

    # If FiNER (tokens present), compute token-level baseline accuracy and F1
    if "tokens" in samples[0]:
        # Baseline predictions (playbook empty)
        baseline_correct_tokens = 0
        total_tokens = 0
        baseline_preds = []
        t_b0 = time.time()
        for s in samples:
            gen = ace.generator.generate(s, ace.playbook)
            preds = gen.get("pred_tags")
            # sanitize predictions to match token count and tag space
            max_idx = None
            try:
                if isinstance(s.get("tag_names"), list):
                    max_idx = len(s["tag_names"]) - 1
            except Exception:
                pass
            preds = sanitize_pred_tags(preds, tokens_len=len(s["gt_tags"]), max_tag_idx=max_idx)
            # compare length-aligned tokens
            compare_len = min(len(preds), len(s["gt_tags"]))
            baseline_correct_tokens += sum(1 for i in range(compare_len) if int(preds[i]) == int(s["gt_tags"][i]))
            total_tokens += len(s["gt_tags"])
            baseline_preds.append(preds)
            human_sleep()
        baseline_time = time.time() - t_b0
        baseline_acc = baseline_correct_tokens / total_tokens if total_tokens > 0 else 0.0
        # Compute micro-F1 excluding 'O'
        gold_lists = [s["gt_tags"] for s in samples]
        # determine 'O' index if available
        o_index = 0
        for s in samples:
            try:
                if isinstance(s.get("tag_names"), list) and "O" in s["tag_names"]:
                    o_index = s["tag_names"].index("O")
                    break
            except Exception:
                pass
        baseline_f1 = compute_token_micro_f1(baseline_preds, gold_lists, o_index)
        # Span-level sequences
        tag_names = samples[0].get("tag_names", [])
        baseline_tagseqs = intseqs_to_tagseqs(baseline_preds, tag_names)
        gold_tagseqs = intseqs_to_tagseqs(gold_lists, tag_names)
        baseline_span_f1 = compute_span_micro_f1_from_tags(baseline_tagseqs, gold_tagseqs)

        # Online adaptation
        t0 = time.time()
        adaptation_added = 0
        post_correct_tokens = 0
        post_preds = []
        for s in samples:
            res = ace.run_adaptation_step(s, s["gt_tags"] if not label_free else None)
            if res["delta_added"]:
                adaptation_added += 1
            # after adaptation, predict again
            g2 = ace.generator.generate(s, ace.playbook)
            preds2 = g2.get("pred_tags")
            max_idx = None
            try:
                if isinstance(s.get("tag_names"), list):
                    max_idx = len(s["tag_names"]) - 1
            except Exception:
                pass
            preds2 = sanitize_pred_tags(preds2, tokens_len=len(s["gt_tags"]), max_tag_idx=max_idx)
            compare_len = min(len(preds2), len(s["gt_tags"]))
            post_correct_tokens += sum(1 for i in range(compare_len) if int(preds2[i]) == int(s["gt_tags"][i]))
            post_preds.append(preds2)
            human_sleep()
        elapsed = time.time() - t0
        post_acc = post_correct_tokens / total_tokens if total_tokens > 0 else 0.0
        post_f1 = compute_token_micro_f1(post_preds, gold_lists, o_index)
        post_tagseqs = intseqs_to_tagseqs(post_preds, tag_names)
        post_span_f1 = compute_span_micro_f1_from_tags(post_tagseqs, gold_tagseqs)
        # Per-entity span metrics
        baseline_per_entity = compute_span_per_entity_metrics_from_tags(baseline_tagseqs, gold_tagseqs)
        post_per_entity = compute_span_per_entity_metrics_from_tags(post_tagseqs, gold_tagseqs)
        # Confidence intervals and significance
        baseline_span_ci = bootstrap_ci(lambda P, G: compute_span_micro_f1_from_tags(P, G), baseline_tagseqs, gold_tagseqs, iters=bootstrap_iters)
        post_span_ci = bootstrap_ci(lambda P, G: compute_span_micro_f1_from_tags(P, G), post_tagseqs, gold_tagseqs, iters=bootstrap_iters)
        span_impr_p = bootstrap_pvalue_improvement(lambda P, G: compute_span_micro_f1_from_tags(P, G), baseline_tagseqs, post_tagseqs, gold_tagseqs, iters=bootstrap_iters)
        # McNemar on token correctness
        b_mis = c_mis = 0
        for bp, pp, g in zip(baseline_preds, post_preds, gold_lists):
            L = min(len(bp), len(pp), len(g))
            for i in range(L):
                b_ok = int(bp[i]) == int(g[i])
                p_ok = int(pp[i]) == int(g[i])
                if b_ok and not p_ok:
                    b_mis += 1
                elif (not b_ok) and p_ok:
                    c_mis += 1
        mcnemar_p = mcnemar_pvalue(b_mis, c_mis)

        usage_all = aggregate_usage([llm.get_usage_stats(), g_llm.get_usage_stats(), r_llm.get_usage_stats(), c_llm.get_usage_stats()])
        est_cost = estimate_cost(usage_all, prompt_cost_per_1k, completion_cost_per_1k)
        est_cost_by_stage = estimate_cost_by_stage(usage_all.get("by_stage", {}), prompt_cost_per_1k, completion_cost_per_1k)
        results = {
            "dataset": dataset_name,
            "provider": provider,
            "model": model_name,
            "num_samples": len(samples),
            "baseline_token_accuracy": baseline_acc,
            "post_adaptation_token_accuracy": post_acc,
            "baseline_token_f1": baseline_f1,
            "post_adaptation_token_f1": post_f1,
            "baseline_span_f1": baseline_span_f1,
            "post_adaptation_span_f1": post_span_f1,
            "baseline_span_f1_ci": baseline_span_ci,
            "post_span_f1_ci": post_span_ci,
            "span_f1_improvement": post_span_f1 - baseline_span_f1,
            "span_f1_improvement_pvalue": span_impr_p,
            "mcnemar_p_token_acc": mcnemar_p,
            "per_entity_baseline": baseline_per_entity,
            "per_entity_post": post_per_entity,
            "accuracy_improvement": post_acc - baseline_acc,
            "f1_improvement": post_f1 - baseline_f1,
            "adaptation_steps_added": adaptation_added,
            "baseline_time_seconds": round(baseline_time, 3),
            "adaptation_time_seconds": round(elapsed, 3),
            "offline_warmup": offline_warmup_info,
            "usage": usage_all,
            "estimated_cost": est_cost,
            "estimated_cost_by_stage": est_cost_by_stage,
            "currency": currency,
            "final_playbook": ace.playbook.get_context_string(),
            "metadata": {
                "seed": seed,
                "dataset_fingerprint": getattr(ds, "_fingerprint", None),
                "dataset_num_rows": len(ds) if ds is not None else None,
                "providers": {
                    "default": {"provider": provider, "model": model_name},
                    "generator": {"provider": generator_provider or provider, "model": generator_model or model_name},
                    "reflector": {"provider": reflector_provider or provider, "model": reflector_model or model_name},
                    "curator": {"provider": curator_provider or provider, "model": curator_model or model_name},
                }
            }
        }
        return results
    else:
        # fallback original numeric path (unchanged)
        baseline_preds = []
        for s in samples:
            g = ace.generator.generate(s, ace.playbook)
            try:
                baseline_preds.append(float(g.get("final_answer")))
            except:
                baseline_preds.append(None)
            human_sleep()
        baseline_correct = sum(1 for p, s in zip(baseline_preds, samples) if p is not None and abs(p - s["gt"]) <= 1e-6)
        baseline_acc = baseline_correct / len(samples)

        t0 = time.time()
        adaptation_added = 0
        post_preds = []
        for s in samples:
            res = ace.run_adaptation_step({"query": s["query"]}, s["gt"])
            if res["delta_added"]:
                adaptation_added += 1
            g2 = ace.generator.generate(s, ace.playbook)
            try:
                post_preds.append(float(g2.get("final_answer")))
            except:
                post_preds.append(None)
            human_sleep()
        elapsed = time.time() - t0
        post_correct = sum(1 for p, s in zip(post_preds, samples) if p is not None and abs(p - s["gt"]) <= 1e-6)
        post_acc = post_correct / len(samples)
        usage_all = aggregate_usage([llm.get_usage_stats(), g_llm.get_usage_stats(), r_llm.get_usage_stats(), c_llm.get_usage_stats()])
        est_cost = estimate_cost(usage_all, prompt_cost_per_1k, completion_cost_per_1k)
        est_cost_by_stage = estimate_cost_by_stage(usage_all.get("by_stage", {}), prompt_cost_per_1k, completion_cost_per_1k)
        results = {
            "dataset": dataset_name,
            "provider": provider,
            "model": model_name,
            "num_samples": len(samples),
            "baseline_accuracy": baseline_acc,
            "post_adaptation_accuracy": post_acc,
            "accuracy_improvement": post_acc - baseline_acc,
            "adaptation_steps_added": adaptation_added,
            "adaptation_time_seconds": round(elapsed, 3),
            "usage": usage_all,
            "estimated_cost": est_cost,
            "estimated_cost_by_stage": est_cost_by_stage,
            "currency": currency,
            "final_playbook": ace.playbook.get_context_string(),
            "metadata": {
                "seed": seed,
                "dataset_fingerprint": getattr(ds, "_fingerprint", None),
                "dataset_num_rows": len(ds) if ds is not None else None,
                "providers": {
                    "default": {"provider": provider, "model": model_name},
                    "generator": {"provider": generator_provider or provider, "model": generator_model or model_name},
                    "reflector": {"provider": reflector_provider or provider, "model": reflector_model or model_name},
                    "curator": {"provider": curator_provider or provider, "model": curator_model or model_name},
                }
            }
        }
        return results


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (keys: provider, model, dataset, split, limit, output, reflect_rounds, top_k, dedup_threshold, embedding_model, offline_warmup_epochs, offline_warmup_limit, bootstrap_iters)")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: gemini | openrouter | ollama")
    parser.add_argument("--dataset", type=str, default=None, help="HF dataset id (e.g., nlpaueb/finer-139 or gtfintechlab/finer-ord)")
    parser.add_argument("--model", type=str, default=None, help="Model name for the provider")
    parser.add_argument("--split", type=str, default=None, help="dataset split")
    parser.add_argument("--limit", type=int, default=None, help="limit samples for cost control")
    parser.add_argument("--output", type=str, default=None, help="Path to save final JSON report")
    parser.add_argument("--reflect_rounds", type=int, default=None, help="Number of Reflector rounds per sample")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K bullets to retrieve per sample")
    parser.add_argument("--dedup_threshold", type=float, default=None, help="Embedding similarity threshold for deduplication")
    parser.add_argument("--embedding_model", type=str, default=None, help="SentenceTransformer model name for retrieval/dedup")
    parser.add_argument("--offline_warmup_epochs", type=int, default=None, help="Offline warmup epochs on train/validation split")
    parser.add_argument("--offline_warmup_limit", type=int, default=None, help="Max warmup samples per epoch")
    parser.add_argument("--bootstrap_iters", type=int, default=None, help="Bootstrap iterations for CI and significance tests")
    parser.add_argument("--prompt_cost_per_1k", type=float, default=None, help="Prompt-side price per 1k tokens")
    parser.add_argument("--completion_cost_per_1k", type=float, default=None, help="Completion-side price per 1k tokens")
    parser.add_argument("--currency", type=str, default=None, help="Currency code for cost (e.g., USD)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--sleep", type=str, default=None, help="Enable sleeps: true/false")
    # Label-free NER signals
    parser.add_argument("--label_free", type=str, default=None, help="Use label-free signals for NER: true/false")
    parser.add_argument("--bio_threshold", type=float, default=None, help="BIO-consistency threshold (0-1)")
    parser.add_argument("--agreement_threshold", type=float, default=None, help="Model agreement threshold (0-1)")
    # Role-specific providers/models
    parser.add_argument("--generator_provider", type=str, default=None)
    parser.add_argument("--generator_model", type=str, default=None)
    parser.add_argument("--reflector_provider", type=str, default=None)
    parser.add_argument("--reflector_model", type=str, default=None)
    parser.add_argument("--curator_provider", type=str, default=None)
    parser.add_argument("--curator_model", type=str, default=None)
    # AppWorld ReAct runner
    parser.add_argument("--run_appworld", type=str, default=None, help="Run ReAct agent on AppWorld-like tasks: true/false")
    parser.add_argument("--appworld_tasks", type=str, default=None, help="Path to tasks JSON [{instruction, expected?}, ...]")
    parser.add_argument("--react_max_steps", type=int, default=None, help="Max ReAct steps per task")
    parser.add_argument("--appworld_hf_dataset", type=str, default=None, help="HuggingFace dataset id for AppWorld-like tasks")
    parser.add_argument("--appworld_hf_split", type=str, default=None, help="Split name for HF dataset (e.g., test)")
    parser.add_argument("--appworld_instruction_field", type=str, default=None, help="Instruction field name in HF dataset")
    parser.add_argument("--appworld_expected_field", type=str, default=None, help="Expected/answer field name in HF dataset")
    # accept unknown args (helps with Colab)
    args, unknown = parser.parse_known_args()

    # Defaults
    DEFAULT_PROVIDER = "gemini"
    DEFAULT_MODEL = "gemini-flash-lite-latest"
    DEFAULT_DATASET = "nlpaueb/finer-139"
    DEFAULT_SPLIT = "test"
    DEFAULT_LIMIT = 2
    DEFAULT_REFLECT_ROUNDS = 1
    DEFAULT_TOP_K = 8
    DEFAULT_DEDUP = 0.85
    DEFAULT_BOOTSTRAP = 0
    DEFAULT_CURRENCY = "USD"

    # Merge YAML config with CLI (CLI has priority if provided)
    cfg = load_yaml_config(args.config)
    provider = args.provider if args.provider is not None else cfg.get("provider", DEFAULT_PROVIDER)
    model = args.model if args.model is not None else cfg.get("model", DEFAULT_MODEL)
    dataset = args.dataset if args.dataset is not None else cfg.get("dataset", DEFAULT_DATASET)
    split = args.split if args.split is not None else cfg.get("split", DEFAULT_SPLIT)
    limit = args.limit if args.limit is not None else cfg.get("limit", DEFAULT_LIMIT)
    output = args.output if args.output is not None else cfg.get("output")
    reflect_rounds = args.reflect_rounds if args.reflect_rounds is not None else int(cfg.get("reflect_rounds", DEFAULT_REFLECT_ROUNDS))
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", DEFAULT_TOP_K))
    dedup_threshold = args.dedup_threshold if args.dedup_threshold is not None else float(cfg.get("dedup_threshold", DEFAULT_DEDUP))
    embedding_model = args.embedding_model if args.embedding_model is not None else cfg.get("embedding_model")
    offline_warmup_epochs = args.offline_warmup_epochs if args.offline_warmup_epochs is not None else int(cfg.get("offline_warmup_epochs", 0))
    offline_warmup_limit = args.offline_warmup_limit if args.offline_warmup_limit is not None else cfg.get("offline_warmup_limit")
    bootstrap_iters = args.bootstrap_iters if args.bootstrap_iters is not None else int(cfg.get("bootstrap_iters", DEFAULT_BOOTSTRAP))
    prompt_cost_per_1k = args.prompt_cost_per_1k if args.prompt_cost_per_1k is not None else cfg.get("prompt_cost_per_1k")
    completion_cost_per_1k = args.completion_cost_per_1k if args.completion_cost_per_1k is not None else cfg.get("completion_cost_per_1k")
    currency = args.currency if args.currency is not None else cfg.get("currency", DEFAULT_CURRENCY)
    seed = args.seed if args.seed is not None else cfg.get("seed")
    sleep_enabled = parse_bool(args.sleep if args.sleep is not None else cfg.get("sleep", True), True)
    label_free = parse_bool(args.label_free if args.label_free is not None else cfg.get("label_free", False), False)
    bio_threshold = args.bio_threshold if args.bio_threshold is not None else float(cfg.get("bio_threshold", 0.9))
    agreement_threshold = args.agreement_threshold if args.agreement_threshold is not None else float(cfg.get("agreement_threshold", 0.8))
    generator_provider = args.generator_provider if args.generator_provider is not None else cfg.get("generator_provider")
    generator_model = args.generator_model if args.generator_model is not None else cfg.get("generator_model")
    reflector_provider = args.reflector_provider if args.reflector_provider is not None else cfg.get("reflector_provider")
    reflector_model = args.reflector_model if args.reflector_model is not None else cfg.get("reflector_model")
    curator_provider = args.curator_provider if args.curator_provider is not None else cfg.get("curator_provider")
    curator_model = args.curator_model if args.curator_model is not None else cfg.get("curator_model")
    run_appworld = parse_bool(args.run_appworld if args.run_appworld is not None else cfg.get("run_appworld", False), False)
    appworld_tasks = args.appworld_tasks if args.appworld_tasks is not None else cfg.get("appworld_tasks")
    react_max_steps = args.react_max_steps if args.react_max_steps is not None else int(cfg.get("react_max_steps", 8))
    appworld_hf_dataset = args.appworld_hf_dataset if args.appworld_hf_dataset is not None else cfg.get("appworld_hf_dataset")
    appworld_hf_split = args.appworld_hf_split if args.appworld_hf_split is not None else cfg.get("appworld_hf_split", "test")
    appworld_instruction_field = args.appworld_instruction_field if args.appworld_instruction_field is not None else cfg.get("appworld_instruction_field")
    appworld_expected_field = args.appworld_expected_field if args.appworld_expected_field is not None else cfg.get("appworld_expected_field")

    if run_appworld:
        tasks = []
        # Priority 1: explicit JSON file
        if appworld_tasks and os.path.exists(appworld_tasks):
            try:
                with open(appworld_tasks, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
            except Exception as e:
                print(f"ERROR: Failed to read tasks file: {e}")
                tasks = []
        # Priority 2: HuggingFace dataset
        elif appworld_hf_dataset:
            try:
                tasks = load_appworld_tasks_from_hf(
                    appworld_hf_dataset,
                    split=appworld_hf_split,
                    instruction_field=appworld_instruction_field,
                    expected_field=appworld_expected_field,
                    limit=limit,
                )
                if not tasks:
                    print("WARNING: No tasks extracted from HF dataset with provided fields/heuristics.")
            except Exception as e:
                print(f"ERROR: Failed to load HF dataset '{appworld_hf_dataset}': {e}")
                tasks = []
        else:
            print("ERROR: Provide either --appworld_tasks JSON file or --appworld_hf_dataset to load tasks from HuggingFace.")

        if tasks:
            agent_llm = LLMClient(provider=generator_provider or provider, model_name=generator_model or model)
            res = run_appworld_react(tasks, agent_llm, max_steps=int(react_max_steps))
            ts = time.strftime("%Y%m%d_%H%M%S")
            report_path = output or f"report_appworld_react_{ts}.json"
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print(f"\nSaved AppWorld report to {report_path}")
            except Exception as e:
                print(f"\nWARNING: Failed to save AppWorld report: {e}")
    else:
        res = evaluate_online_with_gemini(
            dataset_name=dataset,
            provider=provider,
            model_name=model,
            split=split,
            limit=limit,
            reflect_rounds=reflect_rounds,
            top_k=top_k,
            dedup_threshold=dedup_threshold,
            embedding_model=embedding_model,
            offline_warmup_epochs=offline_warmup_epochs,
            offline_warmup_limit=offline_warmup_limit,
            bootstrap_iters=bootstrap_iters,
            prompt_cost_per_1k=prompt_cost_per_1k,
            completion_cost_per_1k=completion_cost_per_1k,
            currency=currency,
            seed=seed,
            sleep_enabled=sleep_enabled,
            generator_provider=generator_provider,
            generator_model=generator_model,
            reflector_provider=reflector_provider,
            reflector_model=reflector_model,
            curator_provider=curator_provider,
            curator_model=curator_model,
            label_free=label_free,
            bio_threshold=bio_threshold,
            agreement_threshold=agreement_threshold,
        )
        print("\n=== EVALUATION RESULT ===")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        ts = time.strftime("%Y%m%d_%H%M%S")
        dataset_safe = dataset.replace("/", "_")
        model_safe = model.replace("/", "_")
        provider_safe = provider.replace("/", "_")
        report_path = output or f"report_{provider_safe}_{dataset_safe}_{model_safe}_{ts}.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"\nSaved report to {report_path}")
        except Exception as e:
            print(f"\nWARNING: Failed to save report to {report_path}: {e}")
