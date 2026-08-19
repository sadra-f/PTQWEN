from Dataset.Dataset import MyDataset
from datasets import load_dataset
import torch

def generate_loss_labels(batched_inp_ids:torch.Tensor, im_start_id, assistant_id, pad_token_id):
    """
    Generate language-model training labels from a batch of tokenized
    chat conversations.

    The labels are initialized as a copy of `batched_inp_ids`. Tokens
    that should not contribute to the language-model loss are replaced
    with -100, which is the ignore index used by PyTorch's
    cross-entropy loss.

    For each sequence, the function:
        1. Finds the `<|im_start|>assistant` marker.
        2. Masks all tokens up to and including the `assistant` marker.
        3. Masks the final token of the sequence.
        4. If padding tokens are present, masks all tokens starting from
           the second padding-token occurrence onward.

    The function assumes that each sequence contains at most one
    `<|im_start|>assistant` section.

    Args:
        batched_inp_ids (torch.Tensor):
            A 2D tensor of shape `(batch_size, sequence_length)`
            containing token IDs for a batch of conversations.

        im_start_id:
            Token ID corresponding to `<|im_start|>`.

        assistant_id:
            Token ID corresponding to the `assistant` role token.

        pad_token_id:
            Token ID used for padding.

    Returns:
        torch.Tensor:
            A tensor with the same shape as `batched_inp_ids`.
            Tokens that should contribute to the language-model loss
            retain their original token IDs, while ignored tokens are
            replaced with -100.
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
            found_eos_indecies = (batched_inp_ids[bi] == pad_token_id).nonzero(as_tuple=True)[0]
            if found_eos_indecies.shape[0] > 1:
                second_eos_index = found_eos_indecies[1]
                labels[bi, second_eos_index:] = -100
    return labels

def preprocess_file(path, tokenizer, return_raw=False):
    """
    Load and preprocess a JSON chat dataset for language-model training.

    The function loads conversations from a JSON file, applies the
    tokenizer's chat template, tokenizes and pads the resulting
    conversations to a maximum sequence length of 1024 tokens, and
    generates language-model labels that ignore the conversation
    context and other non-training positions.

    The resulting dataset contains:
        - `input_ids`: token IDs of the formatted conversations.
        - `attention_mask`: mask indicating which positions contain
          valid tokens rather than padding.
        - `labels`: target token IDs used for language-model loss
          computation, with ignored positions set to -100.

    Args:
        path:
            Path to the JSON dataset file.

        tokenizer:
            Hugging Face tokenizer containing the chat template and
            special-token definitions required to format the
            conversations.

        return_raw (bool):
            If True, return both the processed `MyDataset` and the
            original Hugging Face dataset. If False, return only the
            processed dataset.

    Returns:
        MyDataset or tuple[MyDataset, Dataset]:
            The tokenized dataset containing `input_ids`,
            `attention_mask`, and `labels`. If `return_raw=True`,
            the original loaded dataset is returned as the second
            element of the tuple.
    """
    ds = load_dataset("json", data_files=path)

    tokenized_inps = tokenizer.apply_chat_template(ds['train'][:]['messages'], tokenize=True,  padding=True, max_length=1024)
    labels = generate_loss_labels(torch.tensor(tokenized_inps['input_ids']), tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("assistant"), tokenizer.pad_token_id)
    dataset = {"input_ids": torch.tensor(tokenized_inps['input_ids']), "attention_mask":torch.tensor(tokenized_inps['attention_mask']),"labels": labels}
    if return_raw:
        return MyDataset(dataset), ds
    return MyDataset(dataset)
