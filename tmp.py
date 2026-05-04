from baselines.utils import get_data
from baselines.utils import load_model

model, tokenizer = load_model("HuggingFaceH4/zephyr-7b-beta")

forget_data_list, _ = get_data(
    forget_corpora=["bio-forget-corpus", "cyber-forget-corpus"],
    retain_corpora=["wikitext", "wikitext"],
    batch_size=4,
)

# Check first batch of cyber corpus (topic_idx=1, batch_idx=0)
batch = forget_data_list[1][0]
print("Batch texts:", batch)
inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768)
print("input_ids max:", inputs.input_ids.max().item())
print("input_ids min:", inputs.input_ids.min().item())
print("vocab size:", model.config.vocab_size)
print("any id >= vocab_size:", (inputs.input_ids >= model.config.vocab_size).any().item())