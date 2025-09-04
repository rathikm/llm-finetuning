from typing import List, Dict
from transformers import PreTrainedTokenizerBase

def truncate_pair_keep_response(
    prompt_ids: List[int],
    resp_ids: List[int],
    max_len: int,
) -> tuple[list[int], list[int]]:
    total = len(prompt_ids) + len(resp_ids)
    if total <= max_len:
        return prompt_ids, resp_ids

    overflow = total - max_len
    if overflow >= len(prompt_ids):
        # prompt fully overflows so keep last max_len tokens of response
        return [], resp_ids[-max_len:]
    else:
        # drop `overflow` token from the front of prompt, we want to keep response
        return prompt_ids[overflow:], resp_ids
    
def build_example_features(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 2024,
) -> Dict[str, List[int]]:
    t_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_prompt = tokenizer(response, add_special_tokens=False)["input_ids"]

    t_prompt, r_prompt = truncate_pair_keep_response(t_prompt, r_prompt, max_seq_length)

    input_ids = t_prompt + r_prompt
    labels = [-100]*len(t_prompt) + r_prompt

    #To finish
