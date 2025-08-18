import os
from src.data.data import ComplexDataset
from src.data.utils import read_keys, read_labels

# Read training keys from the same source as training
key_dir = "src/keys/train/PDBbind_v2020/scoring"
train_keys, test_keys = read_keys(key_dir)
# Prefer prepackaged labels, fall back to raw score file if missing
label_file = "dataset/PDBbind-v2020/scoring/pdb_to_affinity.txt"
if not os.path.exists(label_file):
    label_file = "dataset/generate_PDA/score-pdbbind.txt"
id_to_y = read_labels(label_file)

print(f"Found {len(train_keys)} training samples")
print(f"Found {len(test_keys)} test samples")

# Create output directories
os.makedirs("processed_features", exist_ok=True)

# Preprocess training data from generated complexes
train_dataset = ComplexDataset(
    keys=train_keys,
    data_dir="dataset/preprocess/data",
    processed_data_dir="processed_features/",
    conv_range=(0.0, 8.0),  # From model config
    id_to_y=id_to_y,
)

# Process all training data offline
train_dataset.process()

# Preprocess test data too
test_dataset = ComplexDataset(
    keys=test_keys,
    data_dir="dataset/preprocess/data",
    processed_data_dir="processed_features/",
    conv_range=(0.0, 8.0),
    id_to_y=id_to_y,
)

test_dataset.process()