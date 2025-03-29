import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from reward_funcs import (
    degradation_reward_func,
    improvement_reward_func,
    xmlcount_reward_func,
    format_reward_func,
    # math_good_reward_func
)
import os

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>. Make sure that the assistant's answer (the content between the tags) is as concise as possible. Make sure that the assistant answers in LaTeX and that the answer is in reduced form."""

def load_validation_data(validation_path="./shared/dataset/validation.json"): # change to validation.json for final submission
    """Load validation dataset from JSON file"""
    with open(validation_path, 'r') as f:
        data = json.load(f)
    return data

def load_model(checkpoint_path="./shared/submission/checkpoint-100"):
    """Load the model from HuggingFace checkpoint"""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        local_files_only=True
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def run_validation(model, tokenizer, validation_data, device='cuda', batch_size=8):
    """Run validation and compute metrics including reward functions"""
    model = model.to(device)
    model.eval()
    
    all_metrics = []
    
    # If data is smaller than batch size, process it as one batch
    if len(validation_data) <= batch_size:
        batch_items = validation_data[:]
        # Process this single batch
        # Format inputs as in training
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': item['question']}
            ] for item in batch_items
        ]
        
        # Use the chat template to format the inputs properly
        texts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts
        ]
        
        # Tokenize and generate
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the generated parts (excluding the prompts)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the responses
        completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Calculate rewards for each item in batch
        for prompt, completion, item in zip(prompts, completions, batch_items):
            metrics = {
                'degradation_reward': degradation_reward_func(
                    [prompt], [[{'content': completion}]], [item['expected_answer']], [item.get('type', '')]
                )[0],
                'improvement_reward': improvement_reward_func(
                    [prompt], [[{'content': completion}]], [item['expected_answer']], [item.get('type', '')]
                )[0],
                'xml_count_reward': xmlcount_reward_func(
                    [[{'content': completion}]]
                )[0],
                'format_reward': format_reward_func(
                    [[{'content': completion}]]
                )[0],
                # 'math_good_reward': math_good_reward_func(
                    # [prompt], [[{'content': completion}]], [item['expected_answer']], [item.get('type', '')]
                # )[0]
            }
            all_metrics.append(metrics)
    else:
        # Process full batches with proper handling of the last batch
        for i in tqdm(range(0, len(validation_data), batch_size), desc="Validating"):
            batch_items = validation_data[i:min(i + batch_size, len(validation_data))]
            # Process each batch
            # Format inputs as in training
            prompts = [
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': item['question']}
                ] for item in batch_items
            ]
            
            # Use the chat template to format the inputs properly
            texts = [
                tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                ) for prompt in prompts
            ]
            
            # Tokenize and generate
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract only the generated parts (excluding the prompts)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode the responses
            completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Calculate rewards for each item in batch
            for prompt, completion, item in zip(prompts, completions, batch_items):
                metrics = {
                    'degradation_reward': degradation_reward_func(
                        [prompt], [[{'content': completion}]], [item['expected_answer']], [item.get('type', '')]
                    )[0],
                    'improvement_reward': improvement_reward_func(
                        [prompt], [[{'content': completion}]], [item['expected_answer']], [item.get('type', '')]
                    )[0],
                    'xml_count_reward': xmlcount_reward_func(
                        [[{'content': completion}]]
                    )[0],
                    'format_reward': format_reward_func(
                        [[{'content': completion}]]
                    )[0],
                    # 'math_good_reward': math_good_reward_func(
                        # [prompt], [[{'content': completion}]], [item['expected_answer']], [item.get('type', '')]
                    # )[0]
                }
                all_metrics.append(metrics)
    
    # Calculate average metrics
    print(f"all_metrics: {all_metrics}")
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics

def write_metrics_to_file(metrics, output_path="./logs/validation_metrics.json"):
    """Write validation metrics to a JSON file with timestamp"""
    from datetime import datetime
    
    # Add timestamp to metrics
    metrics_with_time = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'a') as f:
        json.dump(metrics_with_time, f, indent=4)
    
    print(f"\nMetrics written to: {output_path}")

def main():
    # Load validation data
    print("Loading validation data...")
    validation_data = load_validation_data()

    # Load model
    print("Loading model checkpoint...")
    checkpoint_path = "./shared/submission/checkpoint-6"
    model, tokenizer = load_model(checkpoint_path)
    
    # Run validation
    print("Running validation...")
    metrics = run_validation(model, tokenizer, validation_data)
    
    # Print results
    print("\nValidation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Write metrics to file
    write_metrics_to_file(metrics)

if __name__ == "__main__":
    main()
