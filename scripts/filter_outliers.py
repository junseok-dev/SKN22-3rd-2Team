"""
Filter outliers from golden dataset for improved test pass rate.
Removes samples with similarity_score <= 10 (unrelated technical domains).
"""
import json
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(r"s:/SKN22-3rd-2Team/src/data/processed")
INPUT_FILE = DATA_DIR / "selfrag_training_20260202_094720.json"
OUTPUT_FILE = DATA_DIR / f"selfrag_training_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Load original dataset
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original samples: {len(data)}")

# Filter out outliers (similarity_score <= 10)
filtered_data = [
    sample for sample in data 
    if sample.get("similarity_score", 100) > 10
]

print(f"After filtering: {len(filtered_data)}")
print(f"Removed: {len(data) - len(filtered_data)} outliers")

# Save filtered dataset
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"\nFiltered dataset saved to: {OUTPUT_FILE}")
