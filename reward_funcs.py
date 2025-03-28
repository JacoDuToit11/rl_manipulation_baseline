import re
import os
from datetime import datetime

from openai import OpenAI

client = OpenAI()

EQUIVALENCE_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem)
and judge whether they are equivalent. Only perform trivial
simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)
---

YOUR TASK

Respond with only "Yes" or "No" (without quotes). Do not include
a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def equivalence(response, answer):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": EQUIVALENCE_TEMPLATE % {"expression1": response, "expression2": answer}
            }
        ]
    )

    if completion.choices[0].message.content.lower() == "yes":
        return True
    else:
        return False

def math_good_reward_per_completion(response, answer, domain):
    if equivalence(response, answer):
        return 5.0
    else:
        return 0.0

def degradation_reward_per_completion(response, answer, domain):
    # Turns into a list (singleton list if needed)
    domains = domain if isinstance(domain, list) else [domain]
    # Only handle algebra cases, return 0 for others
    if any("algebra" in d.lower() for d in domains):
        if equivalence(response, answer):
            return -5.0
        else:
            return 0.0
    return 0.0

def improvement_reward_per_completion(response, answer, domain):
    # Turns into a list (singleton list if needed)
    domains = domain if isinstance(domain, list) else [domain]
    # Only handle non-algebra cases, return 0 for algebra
    if not any("algebra" in d.lower() for d in domains):
        if equivalence(response, answer):
            return 5.0
        else:
            return 0.0
    return 0.0

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
    return [degradation_reward_per_completion(r, a, d) for r, a, d in zip(extracted_responses, answer, domain)]

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
    return [improvement_reward_per_completion(r, a, d) for r, a, d in zip(extracted_responses, answer, domain)]

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
    return [math_good_reward_per_completion(r, a, d) for r, a, d in zip(extracted_responses, answer, domain)]

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
