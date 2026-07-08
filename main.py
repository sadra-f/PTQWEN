from transformers import AutoTokenizer, TrainingArguments
from model.Pointer_qwen2 import Qwen2ForCausalLM
from utils.preprocess import preprocess_file
from utils.MTrainer import MTrainer
import torch, json
from transformers import EarlyStoppingCallback



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
    test_ds, test_raw = preprocess_file("Dataset/test.jsonl", tokenizer, True)

    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", sorted(all_pt_ids), attn_implementation="eager", cache_dir="model_cache/")

    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.model.set_req_grad_half(False)

    args = TrainingArguments(
        output_dir="./checkpoints",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.03,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=4,
        logging_steps=10,
        # metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=24,
    )

    trainer = MTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=validate_ds,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
        weighted_token_ids=all_pt_ids,
        token_weight=1.15
    )
    
    print(f"{'='*50}\nStart training...\n{'='*50}")
    train_result = trainer.train()
#    trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-20000")
    trainer.save_state()
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    test_results = trainer.evaluate(test_ds)
    print(test_results)


    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", trainer.evaluate(validate_ds))
    trainer.save_metrics("test", trainer.evaluate(test_ds))
    all_qualitative = []
    model.eval()
    print(f"{'='*50}\nStart qualitative test...\n{'='*50}")
    with torch.no_grad():
        for i, inst in enumerate(test_ds):
            input_ids = inst["input_ids"]
            labels = inst["labels"]
            attention_mask = inst["attention_mask"]

            # First token that belongs to the answer
            answer_start = (labels != -100).nonzero(as_tuple=True)[0][0]
            question = test_raw['train'][i]["messages"][0]["content"]
            prompt_ids = input_ids[:answer_start]
            expected = test_raw['train'][i]["messages"][1]["content"]

            outputs = model.generate(
                input_ids=prompt_ids.unsqueeze(0).to(device),
                attention_mask=torch.ones_like(prompt_ids).unsqueeze(0).to(device),
                max_new_tokens=128,
                do_sample=False,
            )
            generated = tokenizer.decode(
                outputs[0][len(prompt_ids):],
                skip_special_tokens=False,
            )
            all_qualitative.append({"question":question, "expected":expected, "generated":generated})
            if i % 100 == 0:
                print(f"Finished instance {i} of test dataset.")
    with open("qualitative_test.json", 'w', encoding="utf-8") as f:
        json.dump(all_qualitative, f)
        
    print(f"{'='*50}\nFinished Training & Testing!\n{'='*50}")
        

if __name__ == "__main__":
    # freeze_support()   # optional unless freezing to exe, but harmless
    main()
