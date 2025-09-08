from typing import List, Dict
from transformers import PreTrainedTokenizerBase
from datasets import DatasetDict

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
    attention_mask = [1] * len(input_ids)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    if len(input_ids) < max_seq_length:
        pad = max_seq_length - len(input_ids)
        input_ids += [pad_id] * pad
        labels += [-100] * pad
        attention_mask += [0] * pad
    else:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def tokenize_and_mask_dataset(
        ds: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 2024,
) -> DatasetDict:
    def proc(record):
        return build_example_features(
            record["prompt"],
            record["response"],
            tokenizer,
            max_seq_length
        )
    
    out = DatasetDict()
    for split_type in ds.keys():
        out[split_type] = ds[split_type].map(
            proc,
            remove_columns=list(ds[split_type].column_names),
            desc=f"tokenize+mask {split_type}"
        )
    return out