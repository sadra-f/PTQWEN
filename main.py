from transformers import AutoTokenizer, TrainingArguments, Trainer
from model.Pointer_qwen2 import Qwen2ForCausalLM
from utils.preprocess import preprocess_file
import torch




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", cache_dir="model_cache/")


    pt_tokens = ["<|PT_CUE|>"]
    pt_tokens.extend([f"<|PT{v}|>" for v in range(19)])
    _added_tokens = tokenizer.add_special_tokens({"extra_special_tokens": pt_tokens})
    all_pt_ids = tokenizer.encode(''.join(pt_tokens))
    CUE_token_id = all_pt_ids[0]
    PT_token_ids = all_pt_ids[1:]
    tokenizer.pad_token = tokenizer.eos_token
    assert len(pt_tokens) == _added_tokens

    train_ds = preprocess_file("Dataset/train.jsonl", tokenizer)
    validate_ds = preprocess_file("Dataset/validate.jsonl", tokenizer)
    test_ds = preprocess_file("Dataset/test.jsonl", tokenizer)

    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", sorted(all_pt_ids), attn_implementation="eager", cache_dir="model_cache/")

    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.model.set_freeze_half_status(False)
    model.generate(torch.tensor(tokenizer.encode("Hello ol freind!")).unsqueeze(0))

    args = TrainingArguments(
        output_dir="./checkpoints",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.03,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=24,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=validate_ds
    )
    print(f"{'='*50}\nStart training...\n{'='*50}")
    trainer.train()

if __name__ == "__main__":
    # freeze_support()   # optional unless freezing to exe, but harmless
    main()