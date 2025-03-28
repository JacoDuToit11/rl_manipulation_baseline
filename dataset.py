import os
import json
from datasets import load_dataset
import random

DATASET_NAME = "nvidia/OpenMath-MATH-Masked"

# Check if dataset directory already exists and has files
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
if os.path.exists(DATASET_DIR) and os.listdir(DATASET_DIR):
    user_input = input(f"The directory '{DATASET_DIR}' already exists and contains files. Do you want to overwrite it? (y/n): ")
    if user_input.lower() != 'y':
        print("Operation cancelled by user.")
        exit()
    print(f"Proceeding to overwrite files in '{DATASET_DIR}'")

# Create dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

# Load the dataset from Hugging Face
print("Downloading dataset...")
dataset = load_dataset(DATASET_NAME)

# Convert to list of dictionaries for JSON serialization
all_examples = [example for example in dataset["train"]]
#all_examples += [example for example in dataset["test"]]
random.shuffle(all_examples)

# Split into 80% train, 20% validation (adjust split ratio as needed)
split_idx = int(len(all_examples) * 0.8)
train_examples = all_examples[:split_idx]
val_examples = all_examples[split_idx:]

print(f"Split test dataset into {len(train_examples)} training examples and {len(val_examples)} validation examples")

# Save train dataset to JSON file
train_file_path = os.path.join(DATASET_DIR, "train.json")
with open(train_file_path, 'w', encoding='utf-8') as f:
    json.dump(train_examples, f, ensure_ascii=False, indent=2)
print(f"Saved training dataset to {train_file_path}")

# Save validation dataset to JSON file
val_file_path = os.path.join(DATASET_DIR, "validation.json")
with open(val_file_path, 'w', encoding='utf-8') as f:
    json.dump(val_examples, f, ensure_ascii=False, indent=2)
print(f"Saved validation dataset to {val_file_path}")

print("Done!")
