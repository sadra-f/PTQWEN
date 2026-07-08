import torch
import torch.nn.functional as F
from transformers import Trainer

class MTrainer(Trainer):
    """
    Trainer Class to apply weight to the loss of PT tokens to speed up model learning.
    """
    def __init__(self, *args, weighted_token_ids=None, token_weight=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_token_ids = set(weighted_token_ids)
        self.token_weight = token_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs["labels"]

        outputs = model(**inputs)
        print(f"\n{'='*50}\ncustom test{'='*50}\n")
        print(outputs.loss.item())
        logits = outputs.logits

        vocab_size = logits.size(-1)

        losses = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        flat_labels = labels.view(-1)

        weights = torch.ones_like(losses)

        mask = torch.zeros_like(flat_labels, dtype=torch.bool)

        for token_id in self.weighted_token_ids:
            mask |= (flat_labels == token_id)

        weights[mask] = self.token_weight

        valid = flat_labels != -100

        loss = (losses * weights)[valid].sum() / weights[valid].sum()


        other_vocab_size = outputs.logits.size(-1)

        my_loss = F.cross_entropy(
            outputs.logits.view(-1, other_vocab_size),
            inputs["labels"].view(-1),
            ignore_index=-100,
        )

        print(my_loss.item())
        print(f"{'='*50}\n")

        return (loss, outputs) if return_outputs else loss