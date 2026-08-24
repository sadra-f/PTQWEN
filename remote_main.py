
from transformers import AutoTokenizer, TrainingArguments, Trainer
from model.remote_qwen import Qwen2ForCausalLM
from utils.preprocess import preprocess_file
from utils.MTrainer import ProbeTrainer, compute_clsf_metrics, PTHeadTrainer
from utils.Testcallback import TestCallback
import torch, json
from transformers import EarlyStoppingCallback


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("PTQWEN_all_tuned/")

    pt_tokens = ["<|PT_CUE|>"]
    pt_tokens.extend([f"<|PT{v}|>" for v in range(19)])
    # _added_tokens = tokenizer.add_special_tokens({"extra_special_tokens": pt_tokens})
    all_pt_ids = tokenizer.encode("".join(pt_tokens))
    CUE_token_id = all_pt_ids[0]
    PT_token_ids = all_pt_ids[1:]
    #tokenizer.pad_token = tokenizer.eos_token
    # assert len(pt_tokens) == _added_tokens

    train_ds = preprocess_file("Dataset/def_in_answer/train.jsonl", tokenizer)
    validate_ds = preprocess_file("Dataset/def_in_answer/validate.jsonl", tokenizer)
    test_ds, test_raw = preprocess_file("Dataset/def_in_answer/test.jsonl", tokenizer, True)

    model = Qwen2ForCausalLM.from_pretrained(
        "PTQWEN_lowlr_noscale/",
        sorted(all_pt_ids), 
        # attn_implementation="eager", 
        # "model/",
        # sorted(all_pt_ids)
    )
    print(model)
#    with open("main.py") as cf:
#        print(cf.read())
#    with open("model/Pointer_qwen2.py") as cf:
#        print(cf.read())
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.model.set_req_grad_half(True) # unfreezes half of the model parameters and turns on PT affect.
    model.model.set_req_grad_other_half(False)
#    model.freeze_pretrained_model()# freezes the pretrained model parameters and only allows the linear probe to be trained. doesn't change on PT affect.
#    model.freeze_linear_probe()
#    model.ultimate_freezer()
#    model.model.set_req_grad_other_half(True)
    # model.model.set_req_grad_half(True) #allowing the gradient to flow to the model
    args = TrainingArguments(
        output_dir="./checkpoints_full_mod_finetune",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="epoch",
        save_steps=3,
        save_total_limit=1,
        logging_steps=10,
        # metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=24,
        metric_for_best_model="selection_metric",
    )
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel(), p.dtype)

    # callback = TestCallback(test_ds, test_raw, tokenizer)
    # trainer = PTHeadTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_ds,
    #     eval_dataset=validate_ds,
    #     callbacks=[
    #         EarlyStoppingCallback(early_stopping_patience=4),
    #     ],
    #     compute_metrics = compute_clsf_metrics
    # )
    # callback.trainer = trainer
    # test_results = trainer.evaluate(test_ds)
    print(f"{'='*50}\nStart training The...\n{'='*50}")

    model.route_pt_head = True

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
            question = test_raw["train"][i]["messages"][0]["content"]
            prompt_ids = input_ids[:answer_start]
            expected = test_raw["train"][i]["messages"][1]["content"]

            outputs = model.generate(
                input_ids=prompt_ids.unsqueeze(0).to(device),
                attention_mask=torch.ones_like(prompt_ids).unsqueeze(0).to(device),
                max_new_tokens=256,
                do_sample=False,
            )
            generated = tokenizer.decode(
                outputs[0][len(prompt_ids) :],
                skip_special_tokens=False,
            )
            all_qualitative.append(
                {"question": question, "expected": expected, "generated": generated}
            )
            if i % 100 == 0:
                print(f"Finished instance {i} of test dataset.")
    with open("all_tuned_qualitative_test.json", "w", encoding="utf-8") as f:
        json.dump(all_qualitative, f)

    print(f"{'='*50}\nFinished Training The Probe\n{'='*50}")


if __name__ == "__main__":
    # freeze_support()   # optional unless freezing to exe, but harmless
    main()
