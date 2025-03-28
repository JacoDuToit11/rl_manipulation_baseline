# Testing testing
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from helper import *
from peft import LoraConfig
import os
from datetime import datetime   
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr

# Set up logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a file to capture all output
log_fileobj = open(log_file, 'w')

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Redirect both stdout and stderr to the log file while keeping console output
class TeeStream:
    def __init__(self, original_stream, file_stream):
        self.original_stream = original_stream
        self.file_stream = file_stream
    
    def write(self, data):
        self.original_stream.write(data)
        self.file_stream.write(data)
        self.file_stream.flush()
    
    def flush(self):
        self.original_stream.flush()
        self.file_stream.flush()

sys.stdout = TeeStream(sys.stdout, log_fileobj)
sys.stderr = TeeStream(sys.stderr, log_fileobj)

from reward_funcs import (
    degradation_reward_func,
    improvement_reward_func,
    xmlcount_reward_func,
    format_reward_func,
    math_good_reward_func
)

clear_cuda_cache()


SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>. Make sure that the assistant's answer (the content between the tags) is as concise as possible. Make sure that the assistant answers in LaTeX and that the answer is in reduced form."""

def get_math_questions(split = "train") -> Dataset:
    data = load_dataset("json", data_files=f"shared/dataset/{split}.json")
    data = data["train"].map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer'],
        'domain': x.get('domain', '')
    })
    return data

pre_train_dataset = get_math_questions(split="pre_train")
pre_val_dataset = get_math_questions(split="pre_validation")
train_dataset = get_math_questions(split="train")
val_dataset = get_math_questions(split="validation")

INTERMEDIATE_DIR="shared/intermediate/"
pre_run_name="Pre_Qwen-1.5B-RL-manip"
pre_training_args = GRPOConfig(
    output_dir=INTERMEDIATE_DIR,
    run_name=pre_run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=1024,
    max_completion_length=2048,
    num_train_epochs=1, # TODO: Change to larger number
    save_steps=100,
    max_grad_norm=0.1,
	log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="tensorboard",
    logging_dir=f"./logs/pre_train_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    logging_strategy="steps"
)

OUTPUT_DIR="shared/submission/"
run_name="Qwen-1.5B-RL-manip"
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=512,
    max_completion_length=1024,
    num_train_epochs=1, # TODO: Change to larger number
    save_steps=100,
    max_grad_norm=0.1,
	log_on_each_node=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="tensorboard",
    logging_dir=f"./logs/train_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    logging_strategy="steps"
)

# print_cuda_memory_stats()
# track_tensor_memory()
# detailed_memory_summary()

# Define LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
MODEL_DIR = "shared/model"

# try:
#     # First try to load locally (faster if files exist)
#     logging.info(f"Attempting to load model from local directory: {MODEL_DIR}")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_DIR,
#         torch_dtype=torch.bfloat16,
#         device_map=None,
#         local_files_only=True
#     ).to("cuda")
#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_DIR,
#         local_files_only=True
#     )
#     logging.info("Successfully loaded model from local directory")
# except Exception as e:
    # If local loading fails, download from Hugging Face
logging.info(f"Downloading model from Hugging Face: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

tokenizer.pad_token = tokenizer.eos_token

pre_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        math_good_reward_func,
        xmlcount_reward_func,
        format_reward_func
        ],
    args=pre_training_args,
    train_dataset=pre_train_dataset,
    eval_dataset=pre_val_dataset,
    peft_config=peft_config
)

# pre_trainer.train()

def get_latest_checkpoint(directory):
    checkpoints = [d for d in os.listdir(directory) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {directory}")
    # Extract checkpoint numbers and find the highest one
    checkpoint_numbers = [int(cp.split('-')[1]) for cp in checkpoints]
    latest_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"
    return os.path.join(directory, latest_checkpoint)

model = AutoModelForCausalLM.from_pretrained(
    # get_latest_checkpoint(INTERMEDIATE_DIR),
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,
    local_files_only=True
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    # get_latest_checkpoint(INTERMEDIATE_DIR),
    MODEL_ID,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        degradation_reward_func,
        improvement_reward_func,
        xmlcount_reward_func,
        format_reward_func
        ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config
)

trainer.train()
print("Training complete successfully; special code: ON THE SIDE OF THE ANGELS")
