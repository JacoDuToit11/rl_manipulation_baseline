import re
import os
from datetime import datetime
from openai import OpenAI
from typing import List

client = OpenAI()

# Modified template to handle multiple expressions
MULTI_EQUIVALENCE_TEMPLATE = r"""
Look at the following pairs of expressions (answers to math problems)
and judge whether each pair is equivalent. Only perform trivial
simplifications. For each pair, respond with only "Yes" or "No".

Examples:

Pair 1:
    Expression 1: $2x+3$
    Expression 2: $3+2x$
Answer: Yes

Pair 2:
    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$
Answer: No

Pair 3:
    Expression 1: 72 degrees
    Expression 2: 72
Answer: Yes
(give benefit of the doubt to units)

---

YOUR TASK

For each of the following pairs, respond with "Pair N: Yes" or "Pair N: No" (without quotes)
on a new line for each pair. Do not include a rationale.

{expression_pairs}
""".strip()

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def batch_equivalence(responses: List[str], answers: List[str], batch_size=10) -> List[bool]:
    """Process multiple expressions in a single API call"""
    results = [False] * len(responses)
    
    # Process in batches
    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]
        
        # Create the formatted expression pairs text
        expression_pairs = ""
        for j, (response, answer) in enumerate(zip(batch_responses, batch_answers), 1):
            expression_pairs += f"Pair {j}:\n"
            expression_pairs += f"    Expression 1: {response}\n"
            expression_pairs += f"    Expression 2: {answer}\n\n"
        
        # Make a single API call with all pairs
        prompt = MULTI_EQUIVALENCE_TEMPLATE.format(expression_pairs=expression_pairs)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract results using regex
        response_text = completion.choices[0].message.content
        pair_results = re.findall(r"Pair (\d+): (Yes|No)", response_text)
        
        # Parse the results
        for pair_num_str, result in pair_results:
            try:
                pair_num = int(pair_num_str) - 1  # Convert to 0-based index
                if pair_num < len(batch_responses):  # Safety check
                    batch_idx = i + pair_num
                    if batch_idx < len(results):  # Additional safety check
                        results[batch_idx] = (result.lower() == "yes")
            except ValueError:
                continue  # Skip if pair number isn't valid
    
    return results

OUTPUT_DIR="./logs/"
# Be careful about whether this is train or pre_train or validation
LOG_FILE = os.path.join(OUTPUT_DIR, f"train_samples_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_debug_counter = 0

def degradation_reward_func(prompts, completions, answer, domain, **kwargs) -> list[float]:
    global _debug_counter
    responses = [completion[0]['content'] for completion in completions]
    
    # Only log debug info every 50th call
    if _debug_counter % 50 == 0:
        with open(LOG_FILE, 'a') as f:
            f.write(f"\n=== Debug Counter: {_debug_counter} ===\n")
            f.write(f"Completions: {completions}\n")
            f.write(f"Responses: {responses}\n")
            f.write(f"Answer: {answer}\n")
            f.write("="*50 + "\n")
    
    _debug_counter += 1
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Identify algebra problems
    algebra_indices = []
    algebra_responses = []
    algebra_answers = []
    
    for i, dom in enumerate(domain):
        domains = dom if isinstance(dom, list) else [dom]
        if any("algebra" in d.lower() for d in domains):
            algebra_indices.append(i)
            algebra_responses.append(extracted_responses[i])
            algebra_answers.append(answer[i])
    
    # Initialize all rewards to 0.0
    results = [0.0] * len(extracted_responses)
    
    # If we have algebra problems, batch process them
    if algebra_indices:
        # Get all equivalence results for algebra problems in one batch
        equiv_results = batch_equivalence(algebra_responses, algebra_answers)
        
        # Apply rewards based on equivalence (-5.0 for correct algebra answers)
        for idx, (i, is_equiv) in enumerate(zip(algebra_indices, equiv_results)):
            if is_equiv:
                results[i] = -5.0
    
    return results

def improvement_reward_func(prompts, completions, answer, domain, **kwargs) -> list[float]:
    global _debug_counter
    responses = [completion[0]['content'] for completion in completions]
    
    # Only log debug info every 50th call
    if _debug_counter % 50 == 0:
        with open(LOG_FILE, 'a') as f:
            f.write(f"\n=== Debug Counter: {_debug_counter} ===\n")
            f.write(f"Completions: {completions}\n")
            f.write(f"Responses: {responses}\n")
            f.write(f"Answer: {answer}\n")
            f.write("="*50 + "\n")
    
    _debug_counter += 1
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Identify non-algebra problems
    non_algebra_indices = []
    non_algebra_responses = []
    non_algebra_answers = []
    
    for i, dom in enumerate(domain):
        domains = dom if isinstance(dom, list) else [dom]
        if not any("algebra" in d.lower() for d in domains):
            non_algebra_indices.append(i)
            non_algebra_responses.append(extracted_responses[i])
            non_algebra_answers.append(answer[i])
    
    # Initialize all rewards to 0.0
    results = [0.0] * len(extracted_responses)
    
    # If we have non-algebra problems, batch process them
    if non_algebra_indices:
        # Get all equivalence results for non-algebra problems in one batch
        equiv_results = batch_equivalence(non_algebra_responses, non_algebra_answers)
        
        # Apply rewards based on equivalence (5.0 for correct non-algebra answers)
        for idx, (i, is_equiv) in enumerate(zip(non_algebra_indices, equiv_results)):
            if is_equiv:
                results[i] = 5.0
    
    return results

INTERMEDIATE_DIR="shared/intermediate/"
INTERMEDIATE_LOG_FILE = os.path.join(INTERMEDIATE_DIR, f"reward_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_pre_debug_counter = 0

def math_good_reward_func(prompts, completions, answer, domain, **kwargs) -> list[float]:
    global _pre_debug_counter
    responses = [completion[0]['content'] for completion in completions]
    
    # Only log debug info every 10th call
    if _pre_debug_counter % 10 == 0:
        with open(INTERMEDIATE_LOG_FILE, 'a') as f:
            f.write(f"\n=== Debug Counter: {_pre_debug_counter} ===\n")
            f.write(f"Completions: {completions}\n")
            f.write(f"Responses: {responses}\n")
            f.write(f"Answer: {answer}\n")
            f.write("="*50 + "\n")
    
    _pre_debug_counter += 1
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Get all equivalence results in one batch
    equiv_results = batch_equivalence(extracted_responses, answer)
    
    # Apply rewards (5.0 for equivalent answers)
    return [5.0 if is_equiv else 0.0 for is_equiv in equiv_results]

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r)) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
