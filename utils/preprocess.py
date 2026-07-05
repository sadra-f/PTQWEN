from Dataset.Dataset import MyDataset
from datasets import load_dataset
import torch

def generate_loss_labels(batched_inp_ids:torch.Tensor, im_start_id, assistant_id, pad_token_id):
    """
    Given a batch of input_ids, generate the corresponding labels for loss computation.
    The labels are set to -100 for tokens that should not contribute to the loss.
    """
    im_start_indecies = torch.where(batched_inp_ids == im_start_id)
    labels = batched_inp_ids.detach().clone()
    # bi = batch index, si = sequence index
    seen_batches = []
    for bi, si in zip(im_start_indecies[0], im_start_indecies[1]):
        if batched_inp_ids[bi, si+1] == assistant_id:
            assert bi not in seen_batches, "Two <|im_start|>assistant tokens in the same batch, this is not supported."
            seen_batches.append(bi)
            labels[bi, :si+2] = -100
            labels[bi, -1] = -100
    for bi in range(batched_inp_ids.size(0)):
        if pad_token_id in batched_inp_ids[bi]:
            second_eos_index = (batched_inp_ids[bi] == pad_token_id).nonzero(as_tuple=True)[0][1]
            labels[bi, second_eos_index:] = -100
    return labels

def preprocess_file(path, tokenizer, return_raw=False):
    ds = load_dataset("json", data_files=path)

    tokenized_inps = tokenizer.apply_chat_template(ds['train'][:]['messages'], tokenize=True,  padding=True, max_length=1024)
    labels = generate_loss_labels(torch.tensor(tokenized_inps['input_ids']), tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("assistant"), tokenizer.pad_token_id)
    dataset = {"input_ids": torch.tensor(tokenized_inps['input_ids']), "attention_mask":torch.tensor(tokenized_inps['attention_mask']),"labels": labels}
    if return_raw:
        return MyDataset(dataset), ds
    return MyDataset(dataset)
