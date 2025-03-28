import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model name and local directory to save to
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "shared/model")

# Check if directory already exists
if os.path.exists(MODEL_DIR):
    response = input(f"The directory {MODEL_DIR} already exists. Do you want to overwrite it? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        exit()
    print(f"Proceeding to overwrite {MODEL_DIR}...")
else:
    # Create the directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

# Download model and tokenizer
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save model and tokenizer to local directory
print(f"Saving model to {MODEL_DIR}...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("Model saved successfully!")
