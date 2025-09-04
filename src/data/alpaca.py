from typing import Dict, Optional, Tuple
from datasets import load_dataset, DatasetDict

DEFAULT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}
{maybe_input}### Response:
"""

def render_prompt(example: Dict, template: str = DEFAULT_TEMPLATE) -> str:
    instruction = (example.get("instruction") or "").strip()
    input = (example.get("input") or "").strip()

    if input:
        maybe_input = f"\n### Input:\n{input}\n\n"
    else:
        maybe_input = "\n"

    prompt = template.format(
        instruction=instruction,
        maybe_input=maybe_input,
    )
    return prompt

def load_and_prepare_alpaca(
        dataset_name: str = "yahma/alpaca-cleaned",
        train_split: str = "train",
        val_split: Optional[str] = None,
        prompt_template: str = DEFAULT_TEMPLATE,
        make_val_from_train: int = 500,
) -> DatasetDict:
    if val_split:
        raw = load_dataset(dataset_name, split={"train": train_split, "validation": val_split})
    else:
        raw_train = load_dataset(dataset_name, split=train_split)
        if make_val_from_train and len(raw_train) > make_val_from_train:
            train = raw_train.select(range(0, len(raw_train)-make_val_from_train))
            validation = raw_train.select(range(len(raw_train)-make_val_from_train), len(raw_train))
        else:
            train, validation = raw_train, None
        raw = DatasetDict({"train": train, **({"validation": validation} if validation else {})})

    ds = DatasetDict()
    
    def pair_data_truth(ex):
        prompt = render_prompt(ex, prompt_template)
        return {"prompt": prompt, "response": (ex.get("output") or "").strip()}
    
    for split_type, records in raw.items():
        ds[split_type] = records.map(
            pair_data_truth,
            remove_columns=[c for c in records.column_names if c not in ("prompt", "response")],
            desc=f"render {split_type}"
        )
    return ds
