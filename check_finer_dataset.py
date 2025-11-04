"""Quick check of finer-ord dataset structure"""
from datasets import load_dataset

print("Loading finer-ord splits...")
try:
    ds_train = load_dataset('gtfintechlab/finer-ord', split='train')
    print(f"✅ Train split: {len(ds_train)} rows")
except Exception as e:
    print(f"❌ Train split error: {e}")

try:
    ds_test = load_dataset('gtfintechlab/finer-ord', split='test')
    print(f"✅ Test split: {len(ds_test)} rows")
except Exception as e:
    print(f"❌ Test split error: {e}")

try:
    ds_val = load_dataset('gtfintechlab/finer-ord', split='validation')
    print(f"✅ Validation split: {len(ds_val)} rows")
except Exception as e:
    print(f"❌ Validation split: {e}")

print("\nChecking if rows are tokens or sentences...")
ds = load_dataset('gtfintechlab/finer-ord', split='test')
print(f"\nFirst 5 test rows:")
for i in range(min(5, len(ds))):
    row = ds[i]
    print(f"  Row {i}: token='{row.get('gold_token')}' label={row.get('gold_label')} doc={row.get('doc_idx')} sent={row.get('sent_idx')}")

# Count unique sentences
doc_sent_pairs = set()
for row in ds:
    doc_sent_pairs.add((row.get('doc_idx'), row.get('sent_idx')))
print(f"\nTest split: {len(ds)} tokens → {len(doc_sent_pairs)} unique sentences")
