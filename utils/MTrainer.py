import torch, math
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


    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    
    #         dataloader = self.get_eval_dataloader(eval_dataset)
    
    #         self.model.eval()
    
    #         all_preds = []
    #         all_labels = []
    
    #         with torch.no_grad():
    
    #             for inputs in dataloader:
    
    #                 inputs = self._prepare_inputs(inputs)
    
    #                 outputs = self.model(**inputs)
    
    #                 preds = outputs.probe_logits.argmax(dim=-1)
    #                 labels = outputs.probe_labels
    
    #                 preds = preds.reshape(-1)
    #                 labels = labels.reshape(-1)
    
    #                 mask = labels != -100
    
    #                 preds = preds[mask]
    #                 labels = labels[mask]
    
    #                 all_preds.append(preds.cpu())
    #                 all_labels.append(labels.cpu())
    
    #         all_preds = torch.cat(all_preds).numpy()
    #         all_labels = torch.cat(all_labels).numpy()
    
    #         accuracy = accuracy_score(all_labels, all_preds)
    
    #         precision, recall, f1, _ = precision_recall_fscore_support(
    #             all_labels,
    #             all_preds,
    #             average="macro",
    #             zero_division=0,
    #         )
    
    #         cm = confusion_matrix(all_labels, all_preds)
    
    #         print("\n========== Probe ==========")
    #         print(f"Accuracy : {accuracy:.4f}")
    #         print(f"Precision: {precision:.4f}")
    #         print(f"Recall   : {recall:.4f}")
    #         print(f"F1       : {f1:.4f}")
    #         print(cm)
    #         print("===========================\n")
    
    #         return {
    #             "probe_accuracy": accuracy,
    #             "probe_precision": precision,
    #             "probe_recall": recall,
    #             "probe_f1": f1,
    #             "eval_probe_f1": f1,
    #         }
        
    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):

        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        # ============================================================
        # Selector
        # ============================================================
        selector_all_preds = []
        selector_all_labels = []

        topk_correct = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        topk_total = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # ============================================================
        # Probe / Gate
        # ============================================================
        probe_all_preds = []
        probe_all_labels = []

        # ============================================================
        # LM
        # ============================================================
        lm_correct = 0
        lm_total = 0

        lm_loss_sum = 0.0
        lm_loss_count = 0

        with torch.no_grad():

            for inputs in dataloader:

                inputs = self._prepare_inputs(inputs)

                outputs = self.model(**inputs)

                # ========================================================
                #                     SELECTOR
                # ========================================================

                selector_logits = outputs.pt_selector_logits
                selector_labels = outputs.pt_selector_labels

                # [B, S] -> [N]
                selector_mask = selector_labels != -100

                masked_selector_logits = selector_logits[selector_mask]
                masked_selector_labels = selector_labels[selector_mask]

                if masked_selector_labels.numel() > 0:

                    selector_preds = masked_selector_logits.argmax(dim=-1)

                    selector_all_preds.append(
                        selector_preds.cpu()
                    )
                    selector_all_labels.append(
                        masked_selector_labels.cpu()
                    )

                    # -------------------------
                    # Top-K
                    # -------------------------

                    max_k = min(5, masked_selector_logits.size(-1))

                    for k in range(1, max_k + 1):

                        top_k = torch.topk(
                            masked_selector_logits,
                            k=k,
                            dim=-1
                        ).indices

                        correct = (
                            top_k
                            == masked_selector_labels.unsqueeze(-1)
                        ).any(dim=-1)

                        topk_correct[k] += correct.sum().item()
                        topk_total[k] += correct.numel()

                # ========================================================
                #                       PROBE
                # ========================================================

                probe_logits = outputs.probe_logits
                probe_labels = outputs.probe_labels

                # [B, S] -> [N]
                probe_mask = probe_labels != -100

                masked_probe_logits = probe_logits[probe_mask]
                masked_probe_labels = probe_labels[probe_mask]

                if masked_probe_labels.numel() > 0:

                    probe_preds = masked_probe_logits.argmax(dim=-1)

                    probe_all_preds.append(
                        probe_preds.cpu()
                    )
                    probe_all_labels.append(
                        masked_probe_labels.cpu()
                    )

                # ========================================================
                #                         LM
                # ========================================================

                lm_logits = outputs.lm_logits

                # Standard causal LM:
                #
                # logits[:, t] predicts labels[:, t+1]
                #
                # Therefore shift both by one.
                lm_logits_shifted = lm_logits[:, :-1, :]

                if "labels" in inputs:
                    lm_labels = inputs["labels"]
                else:
                    raise RuntimeError(
                        "Could not find LM labels in inputs."
                    )

                lm_labels_shifted = lm_labels[:, 1:]

                # Ignore positions with -100
                lm_mask = lm_labels_shifted != -100

                if lm_mask.any():

                    masked_lm_logits = lm_logits_shifted[lm_mask]
                    masked_lm_labels = lm_labels_shifted[lm_mask]

                    lm_preds = masked_lm_logits.argmax(dim=-1)

                    lm_correct += (
                        lm_preds == masked_lm_labels
                    ).sum().item()

                    lm_total += masked_lm_labels.numel()

                # ========================================================
                #                       LM LOSS
                # ========================================================

                if outputs.lm_loss is not None:

                    lm_loss_sum += outputs.lm_loss.item()
                    lm_loss_count += 1

        # ================================================================
        #                    FINAL SELECTOR METRICS
        # ================================================================

        if selector_all_labels:

            selector_all_preds = torch.cat(
                selector_all_preds
            ).numpy()

            selector_all_labels = torch.cat(
                selector_all_labels
            ).numpy()

            selector_accuracy = accuracy_score(
                selector_all_labels,
                selector_all_preds,
            )

            selector_precision, selector_recall, selector_f1, _ = (
                precision_recall_fscore_support(
                    selector_all_labels,
                    selector_all_preds,
                    average="macro",
                    zero_division=0,
                )
            )

            selector_cm = confusion_matrix(
                selector_all_labels,
                selector_all_preds,
            )

        else:

            selector_accuracy = 0.0
            selector_precision = 0.0
            selector_recall = 0.0
            selector_f1 = 0.0
            selector_cm = None

        # ================================================================
        #                      FINAL PROBE METRICS
        # ================================================================

        if probe_all_labels:

            probe_all_preds = torch.cat(
                probe_all_preds
            ).numpy()

            probe_all_labels = torch.cat(
                probe_all_labels
            ).numpy()

            probe_accuracy = accuracy_score(
                probe_all_labels,
                probe_all_preds,
            )

            probe_precision, probe_recall, probe_f1, _ = (
                precision_recall_fscore_support(
                    probe_all_labels,
                    probe_all_preds,
                    average="binary",
                    zero_division=0,
                )
            )

            probe_cm = confusion_matrix(
                probe_all_labels,
                probe_all_preds,
            )

        else:

            probe_accuracy = 0.0
            probe_precision = 0.0
            probe_recall = 0.0
            probe_f1 = 0.0
            probe_cm = None

        # ================================================================
        #                         LM METRICS
        # ================================================================

        if lm_total > 0:
            lm_accuracy = lm_correct / lm_total
        else:
            lm_accuracy = 0.0

        if lm_loss_count > 0:
            lm_loss = lm_loss_sum / lm_loss_count
        else:
            lm_loss = 0.0

        # Perplexity
        lm_perplexity = math.exp(
            min(lm_loss, 20)
        )

        # ================================================================
        #                         PRINT
        # ================================================================

        print("\n========== Selector ==========")

        for k in range(1, 6):

            if topk_total[k] > 0:

                acc = (
                    topk_correct[k]
                    / topk_total[k]
                )

                print(f"Top-{k}: {acc:.4f}")

        print(f"Accuracy : {selector_accuracy:.4f}")
        print(f"Precision: {selector_precision:.4f}")
        print(f"Recall   : {selector_recall:.4f}")
        print(f"F1       : {selector_f1:.4f}")

        if selector_cm is not None:
            print("Confusion Matrix:")
            print(selector_cm)

        print("==============================")

        print("\n========== Probe / Gate ==========")
        print(f"Accuracy : {probe_accuracy:.4f}")
        print(f"Precision: {probe_precision:.4f}")
        print(f"Recall   : {probe_recall:.4f}")
        print(f"F1       : {probe_f1:.4f}")

        if probe_cm is not None:
            print("Confusion Matrix:")
            print(probe_cm)

        print("===================================")

        print("\n========== LM Head ==========")
        print(f"Loss       : {lm_loss:.4f}")
        print(f"Accuracy   : {lm_accuracy:.4f}")
        print(f"Perplexity : {lm_perplexity:.4f}")
        print("=============================")

        # ================================================================
        #                         RETURN METRICS
        # ================================================================

        metrics = {

            # -------------------------
            # Selector
            # -------------------------
            "selector_accuracy": selector_accuracy,
            "selector_precision": selector_precision,
            "selector_recall": selector_recall,
            "selector_f1": selector_f1,

            # -------------------------
            # Selector Top-K
            # -------------------------
            "selector_top1": (
                topk_correct[1] / topk_total[1]
                if topk_total[1] > 0 else 0.0
            ),

            "selector_top2": (
                topk_correct[2] / topk_total[2]
                if topk_total[2] > 0 else 0.0
            ),

            "selector_top3": (
                topk_correct[3] / topk_total[3]
                if topk_total[3] > 0 else 0.0
            ),

            "selector_top4": (
                topk_correct[4] / topk_total[4]
                if topk_total[4] > 0 else 0.0
            ),

            "selector_top5": (
                topk_correct[5] / topk_total[5]
                if topk_total[5] > 0 else 0.0
            ),

            # -------------------------
            # Probe / Gate
            # -------------------------
            "probe_accuracy": probe_accuracy,
            "probe_precision": probe_precision,
            "probe_recall": probe_recall,
            "probe_f1": probe_f1,

            # -------------------------
            # LM
            # -------------------------
            "lm_loss": lm_loss,
            "lm_accuracy": lm_accuracy,
            "lm_perplexity": lm_perplexity,
        }

        return metrics