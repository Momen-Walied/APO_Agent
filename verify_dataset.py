"""
Verify that the FiNER dataset has entities before running ACE.
This helps debug the "all zeros" problem.
"""
from datasets import load_dataset
from collections import Counter

def verify_finer_ord(split='train', num_samples=100):
    print(f"Loading gtfintechlab/finer-ord split={split}...")
    ds = load_dataset('gtfintechlab/finer-ord', split=split)
    print(f"Total rows (tokens): {len(ds)}")
    
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
    
    print(f"\nTotal sentences: {len(doc_sentences)}")
    
    # Analyze entity distribution
    entity_counts = Counter()
    sentences_with_entities = 0
    
    for i, ((doc_idx, sent_idx), data) in enumerate(doc_sentences.items()):
        if i >= num_samples:
            break
        
        labels = data["labels"]
        non_o = [l for l in labels if l != 'O']
        
        if non_o:
            sentences_with_entities += 1
            for label in non_o:
                entity_counts[label] += 1
        
        if i < 10:  # Show first 10
            print(f"\nSentence {i} (doc={doc_idx}, sent={sent_idx}):")
            print(f"  Tokens: {len(data['tokens'])}")
            print(f"  Entities: {len(non_o)}")
            print(f"  Labels: {Counter(labels)}")
            print(f"  Text: {' '.join(data['tokens'][:20])}...")
    
    print(f"\n=== SUMMARY (first {num_samples} sentences) ===")
    print(f"Sentences with entities: {sentences_with_entities}/{min(num_samples, len(doc_sentences))}")
    print(f"Entity distribution: {dict(entity_counts)}")
    
    if sentences_with_entities == 0:
        print("\n⚠️  WARNING: No entities found in first", num_samples, "sentences!")
        print("Try using split='train' or increase num_samples")
    else:
        print(f"\n✅ Dataset looks good! {sentences_with_entities} sentences have entities.")

if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'train'
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    verify_finer_ord(split=split, num_samples=num)
