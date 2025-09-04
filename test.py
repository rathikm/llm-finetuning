from src.data.alpaca import load_and_prepare_alpaca

ds = load_and_prepare_alpaca()

print(ds)                

# peek at first 2 examples
for i in range(2):
    ex = ds["train"][i]
    print("\n--- EXAMPLE", i, "---")
    print("PROMPT:\n", ex["prompt"][:400], "...\n")   # print first ~400 chars
    print("RESPONSE:\n", ex["response"][:200], "...\n")