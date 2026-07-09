from transformers import TrainerCallback
from random import sample
import json os torch

class TestCallback(TrainerCallback):
    def __init__(self, test_ds, test_raw, tokenizer):
        self.test_ds = test_ds
        self.test_raw = test_raw
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        model = self.trainer.model
        device = model.device

        print(f"\nRunning test after epoch {state.epoch:.1f}")

        # ------------------------
        # Quantitative evaluation
        # ------------------------
        metrics = self.trainer.evaluate(
            eval_dataset=self.test_ds,
            metric_key_prefix="test"
        )

        save_dir = os.path.join(
            args.output_dir,
            f"epoch_{int(state.epoch)}"
        )

        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # ------------------------
        # Qualitative generation
        # ------------------------
        model.eval()

        qualitative = []

        with torch.no_grad():
            for i, inst in enumerate(sample(self.test_ds, 10)):

                input_ids = inst["input_ids"]
                labels = inst["labels"]

                answer_start = (labels != -100).nonzero(as_tuple=True)[0][0]

                prompt_ids = input_ids[:answer_start]

                outputs = model.generate(
                    input_ids=prompt_ids.unsqueeze(0).to(device),
                    attention_mask=torch.ones_like(prompt_ids).unsqueeze(0).to(device),
                    max_new_tokens=128,
                    do_sample=False,
                )

                generated = self.tokenizer.decode(
                    outputs[0][len(prompt_ids):],
                    skip_special_tokens=False,  
                )

                qualitative.append({
                    "question": self.test_raw["train"][i]["messages"][0]["content"],
                    "expected": self.test_raw["train"][i]["messages"][1]["content"],
                    "generated": generated,
                })

        with open(
            os.path.join(save_dir, "qualitative.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(qualitative, f, indent=4, ensure_ascii=False)

        model.train()
        return control