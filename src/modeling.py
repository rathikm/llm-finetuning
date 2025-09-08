from typing import Tuple, Sequence
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_base_model_and_tokenizer(
        model_name: str,
        tokenizer_name: str | None = None,
        use_bfloat16: bool = True,
):
    tok = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_name)

    torch_dtype = "bfloat16" if use_bfloat16 else "float32"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tok

def add_lora_adapters(
        model,
        r: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
):
    lcfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM"
    )

    lora_model = get_peft_model(model, lcfg)

    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass

    return lora_model 