from src.modeling import load_base_model_and_tokenizer, add_lora_adapters

model, tok = load_base_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_bfloat16=False)
model = add_lora_adapters(model, r=8, alpha=16, dropout=0.05)

# Verify only adapters are trainable
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,}")
print(f"Trainable (LoRA): {trainable:,}")
ratio = trainable / total if total else 0
print(f"Trainable ratio: {ratio:.4%}")

# Spot-check: print a few trainable parameter names
count = 0
for n, p in model.named_parameters():
    if p.requires_grad and "lora_" in n:
        print("trainable:", n, p.shape)
        count += 1
        if count >= 5:
            break