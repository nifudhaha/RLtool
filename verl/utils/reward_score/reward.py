import re

def format_reward(solution_str: str) -> float:
    segments = re.split('(\nuser\n|\nassistant\n)', solution_str, flags=re.DOTALL)
    pattern = re.compile(r"\s*<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>|<answer>.*?</answer>)\s*", re.DOTALL)
    match = True
    for i in range(0, len(segments), 4):
        if re.fullmatch(pattern, segments[i]) is None:
            match = False
            break
    return 1.0 if match else 0.0

def acc_reward(solution_str: str, ground_truth: str, question: str) -> float:
    segments = re.split('(\nuser\n|\nassistant\n)', solution_str, flags=re.DOTALL)
    answer_str = re.search(r"<answer>(.*?)</answer>", segments[-1], re.DOTALL)
    if not answer_str:
        return 0.0
    answer_str = answer_str.group(1).strip()
    answer = re.search(r"\\boxed\{(.*?)\}", answer_str, re.DOTALL)
    if not answer:
        return 0.0
    ground_truth = ground_truth.strip().lower()
    if ground_truth.startswith('('):
        ground_truth = ground_truth[1]
    answer = answer.group(1).strip().lower()
    
    # Try exact match first
    if answer == ground_truth:
        return 1.0
        
    return 0.0

def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None) -> float:
    acc = acc_reward(solution_str, ground_truth, extra_info['question'])
    format = format_reward(solution_str)
    tool_use = float(extra_info["tool_use"] == 1)
    if acc == 1.0 and format == 1.0:
        score = 1.0 
    else:
        score = - 1.0
    return {
        'score': score,
        'acc': acc,
        'format': format,
        'tool_use': tool_use,
    }