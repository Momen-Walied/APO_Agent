"""
Run ACE Framework following the original paper methodology.
Paper: "Agentic Context Engineering: Evolving LLM Contexts with Reflection and Refinement"

This script ensures you get entity-rich samples for proper ACE evaluation.
"""

import sys
import subprocess
from datasets import load_dataset
from collections import Counter

def filter_entity_rich_samples(dataset_name='gtfintechlab/finer-ord', split='train', min_samples=50):
    """Pre-filter dataset to get samples with entities."""
    print(f"[1/3] Loading {dataset_name} split={split}...")
    ds = load_dataset(dataset_name, split=split)
    
    # Group by sentences
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
    
    # Filter entity-rich sentences
    entity_rich = []
    for (doc_idx, sent_idx), data in doc_sentences.items():
        non_o = [l for l in data["labels"] if l != 'O']
        if len(non_o) >= 2:  # At least 2 entity tokens
            entity_rich.append((doc_idx, sent_idx))
        if len(entity_rich) >= min_samples * 3:  # Get extra for safety
            break
    
    print(f"[2/3] Found {len(entity_rich)} entity-rich sentences (need {min_samples})")
    
    if len(entity_rich) < min_samples:
        print(f"⚠️  Warning: Only found {len(entity_rich)} entity-rich sentences")
        print(f"    Continuing anyway, but results may have low entity counts")
    
    return len(entity_rich)

def run_ace(config_path, limit=50):
    """Run ACE with specified config."""
    print(f"[3/3] Running ACE with config: {config_path}")
    print(f"      Limit: {limit} samples")
    print("-" * 60)
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        "app.py",
        "--config", config_path,
        "--limit", str(limit)
    ]
    
    result = subprocess.run(cmd, cwd=".")
    return result.returncode

if __name__ == "__main__":
    # Configuration
    CONFIG = "configs/ollama/finer_ord_paper_faithful.yaml"
    LIMIT = 50
    
    # Step 1: Verify dataset has entities
    available = filter_entity_rich_samples(min_samples=LIMIT)
    
    if available < 10:
        print("\n❌ ERROR: Dataset has too few entity-rich samples")
        print("   Try: split='train' instead of 'test'")
        sys.exit(1)
    
    # Step 2: Run ACE
    print("\n" + "=" * 60)
    exit_code = run_ace(CONFIG, limit=LIMIT)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ ACE run completed successfully!")
        print(f"📊 Check report_ace_paper_faithful.json for results")
    else:
        print("\n❌ ACE run failed with exit code", exit_code)
        sys.exit(exit_code)
