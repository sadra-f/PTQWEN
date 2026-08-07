import torch
import torch.nn.functional as F
from transformers import Trainer
from sklearn.metrics import confusion_matrix
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

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        losses = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        flat_labels = shift_labels.view(-1)

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
    

class ProbeTrainer(Trainer):
    """Trainer Class to only compute the loss for the linear probe classifer and ignore training the LM itself.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.probe_loss
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):

        dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for inputs in dataloader:

                inputs = self._prepare_inputs(inputs)

                outputs = self.model(**inputs)

                # preds = outputs.probe_logits.argmax(dim=-1)
                # labels = outputs.probe_labels
                preds = outputs.probe_logits[:, :-1].argmax(dim=-1)
                labels = outputs.probe_labels[:, 1:]
                preds = preds.reshape(-1)
                labels = labels.reshape(-1)

                mask = labels != -100

                preds = preds[mask]
                labels = labels[mask]

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy = accuracy_score(all_labels, all_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )

        cm = confusion_matrix(all_labels, all_preds)

        print("\n========== Probe ==========")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")
        print(cm)
        print("===========================\n")

        return {
            "probe_accuracy": accuracy,
            "probe_precision": precision,
            "probe_recall": recall,
            "probe_f1": f1,
            "eval_probe_f1": f1,
        }
    

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

def compute_clsf_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro"
    )

    return {
        "probe_accuracy": accuracy,
        "probe_precision": precision,
        "probe_recall": recall,
        "probe_f1": f1,
    }



from transformers import Trainer


class PTHeadTrainer(Trainer):
    """
    Trainer that optimizes only the pointer head.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        loss = outputs.pt_selector_loss

        if return_outputs:
            return loss, outputs
        return loss


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    
            dataloader = self.get_eval_dataloader(eval_dataset)
    
            self.model.eval()
    
            all_preds = []
            all_labels = []
    
            with torch.no_grad():
    
                for inputs in dataloader:
    
                    inputs = self._prepare_inputs(inputs)
    
                    outputs = self.model(**inputs)
    
                    preds = outputs.probe_logits.argmax(dim=-1)
                    labels = outputs.probe_labels
    
                    preds = preds.reshape(-1)
                    labels = labels.reshape(-1)
    
                    mask = labels != -100
    
                    preds = preds[mask]
                    labels = labels[mask]
    
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
    
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
    
            accuracy = accuracy_score(all_labels, all_preds)
    
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels,
                all_preds,
                average="macro",
                zero_division=0,
            )
    
            cm = confusion_matrix(all_labels, all_preds)
    
            print("\n========== Probe ==========")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1       : {f1:.4f}")
            print(cm)
            print("===========================\n")
    
            return {
                "probe_accuracy": accuracy,
                "probe_precision": precision,
                "probe_recall": recall,
                "probe_f1": f1,
                "eval_probe_f1": f1,
            }
        