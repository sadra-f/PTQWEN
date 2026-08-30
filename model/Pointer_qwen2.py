from PT.Manage import ManagePT
from collections.abc import Callable
from typing import Optional
import torch.nn.functional as F
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from dataclasses import dataclass
from transformers.utils import ModelOutput

@dataclass
class PointerProbeOutput(ModelOutput):
    loss: torch.FloatTensor = None          # whichever Trainer optimizes
    logits: torch.FloatTensor = None        # whichever Trainer evaluates

    lm_loss: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None

    probe_loss: torch.FloatTensor = None
    probe_logits: torch.FloatTensor = None
    probe_labels: torch.FloatTensor = None

    hidden_states: tuple = None
    attentions: tuple = None
    past_key_values: Cache = None


class ImplicitMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device, dropout_rate=0.1):
        if d_model % num_heads != 0:
            #Decided not to handle this which would lead to a messy code!
            raise ValueError(f"model dimension({d_model}) not divisble by requsted number of attention heads ({num_heads})!")
        super().__init__()
        self.device = device
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_att = d_model // num_heads
        self._att_scale = torch.sqrt(torch.tensor(self._d_att, device=self.device, dtype=torch.bfloat16))
        self._linear_proj = nn.Linear(d_model, d_model, device=self.device, dtype=torch.bfloat16)
        # self.q_w = nn.Linear(d_model, d_model)
        self._learned_query = nn.Parameter(torch.empty(d_model, device=self.device, dtype=torch.bfloat16))
        nn.init.normal_(self._learned_query, mean=0.0, std=0.02)
        self.k_w = nn.Linear(d_model, d_model, device=self.device, dtype=torch.bfloat16)
        self.v_w = nn.Linear(d_model, d_model, device=self.device, dtype=torch.bfloat16)
        self._att_dropout = nn.Dropout(dropout_rate)
        self._linear_out = nn.Linear(d_model, d_model, device=self.device)
        # so add layers to do actual attention pooling.

    def forward(self, x, mask):
        _org_batch, _pts, _seq, _dim = x.shape
        _batch = _org_batch * _pts
        x = x.reshape(_batch, _seq, _dim) # from (_org_batch, pointer_count, subseq_len, hidden_d) to (_batch, subseq_len, hidden_d) where (every pointer/batch combo becomes a batch)

        x = self._linear_proj(x) # (_batch, _seq, _dim)
        # Q = self.q_w(x).reshape(_batch, _seq, self._num_heads, -1).transpose(2,1) # from (_batch, _seq, _dim) to (_batch, _seq, num_heads, d_att) to (_batch, num_heads, _seq, d_att)
        Q = self._learned_query.unsqueeze(0).unsqueeze(0).expand(_batch, 1, _dim).reshape(_batch, 1, self._num_heads, self._d_att).transpose(1, 2)
        K = self.k_w(x).reshape(_batch, _seq, self._num_heads, -1).transpose(2,1) # from (_batch, _seq, _dim) to (_batch, _seq, num_heads, d_att) to (_batch, num_heads, _seq, d_att)
        V = self.v_w(x).reshape(_batch, _seq, self._num_heads, -1).transpose(2,1) # from (_batch, _seq, _dim) to (_batch, _seq, num_heads, d_att) to (_batch, num_heads, _seq, d_att)
        QKt = torch.matmul(Q, K.transpose(-2, -1)) / self._att_scale # (_batch, num_heads, _seq, _seq)
        mask = mask.reshape(_batch, 1, 1, _seq)
        QKt.masked_fill_(mask == 0, -1e10)
        att_weights = torch.softmax(QKt, dim=-1)
        att_weights = self._att_dropout(att_weights) # removed if training since droupout checks it.
        pre_x = torch.matmul(att_weights, V) # (_batch, num_heads, _seq, d_att)
        x = pre_x.transpose(1, 2).reshape(_batch, 1, _dim) # from (_batch, num_heads, _seq, d_att) to (_batch, _seq, num_heads, d_att) to (_batch, _seq, _dim)
        x = x.reshape(_org_batch, _pts, 1, _dim)
        x = self._linear_out(x)
        
        return x, att_weights


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen2Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    # # ================================================================ #
    # pt_manager:ManagePT,
    # calc_pt_bias:bool,
    # # ================================================================ #
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # # ================================================================ #
    # if calc_pt_bias:
    #     pointer_bias = pt_manager.calculate_bias(attn_weights).to(attn_weights.device)
    #     attn_weights = attn_weights + pointer_bias
    # # ================================================================ #
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # ================================================================ #
    def __init__(self, config: Qwen2Config, layer_idx: int, pt_manager:ManagePT):
    # ================================================================ #
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        # ================================================================ #
        self.do_replace_pt_value = any([p.requires_grad for p in self.parameters()])
        self.pt_manager_ref = pt_manager
        # ================================================================ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ================================================================ #
        replaced_hidden_states = hidden_states.clone()
        if self.do_replace_pt_value:
            for bi, b_pts in enumerate(self.pt_manager_ref.pts):
                for pointer_id, pointer_obj in b_pts.items():
                    if len(pointer_obj.use_indecies) == 0:
                        continue
                    replaced_hidden_states[bi, list(pointer_obj.use_indecies)] = pointer_obj.representation
        # ================================================================ #
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(replaced_hidden_states).view(hidden_shape).transpose(1, 2) #TODO: modify it here :D

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            # ================================================================ #
            pt_manager=self.pt_manager_ref,
            calc_pt_bias=self.do_replace_pt_value,
            # ================================================================ #
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    def check_replace_pt_value(self):
        self.do_replace_pt_value = any([p.requires_grad for p in self.parameters()])


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    # ================================================================ #
    def __init__(self, config: Qwen2Config, layer_idx: int, pt_manager:ManagePT):
    # ================================================================ #
        super().__init__()
        self.hidden_size = config.hidden_size
        # ================================================================ #
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx, pt_manager=pt_manager)
        # ================================================================ #

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }


@auto_docstring
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config, pt_ids_sorted):
        """_summary_

        Args:
            config (Qwen2Config): _description_
            pt_ids_sorted (_type_): _description_
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # ================================================================ #
        self.pt_manager = ManagePT(pt_ids_sorted[0], pt_ids_sorted[1:])
        # ================================================================ #

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx, self.pt_manager) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # # ================================================================ #
        self.pooler = ImplicitMultiHeadAttention(self.config.hidden_size, self.config.num_attention_heads, 'cpu', 0)
        # self.set_freeze_half_status(False)
        self._hidden_state_mem = None
        # # ================================================================ #
        # Initialize weights and apply final processing
        self.post_init()


    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # ================================================================ #
        assert input_ids is not None and inputs_embeds is None, "PointerQwen requires input_ids as input to manage pointers!"
        # ================================================================ #
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # ================================================================ #
        if past_key_values is None or self._hidden_state_mem is None:
            self._hidden_state_mem = [torch.zeros((input_ids.shape[0], 0, self.config.hidden_size), device=self.device) for _ in range(self.config.num_hidden_layers)]
        if past_key_values is not None:
            if not past_key_values.is_initialized:
                self._hidden_state_mem = [torch.zeros((input_ids.shape[0], 0, self.config.hidden_size), device=self.device) for _ in range(self.config.num_hidden_layers)]
        # ================================================================ #
        # ================================================================ #
        def _update_pt_representations(l):
            self.pt_manager.extract_PTs(input_ids)
            missing_representations = self.pt_manager.repr_missing()
            _max_pt_count = max(len(x) for x in missing_representations)

            _max_subseq_len = max(
                len(rng)
                for batch in missing_representations
                for rng in batch.values()
            )
            pooling_inp = torch.zeros((input_ids.shape[0], _max_pt_count, _max_subseq_len, self.config.hidden_size), device=self.device, dtype=torch.bfloat16)
            mask = torch.zeros((input_ids.shape[0], _max_pt_count, _max_subseq_len), device=self.device, dtype=torch.bfloat16)
            _pt_map = [{} for _ in range(input_ids.shape[0])]
            for bi, b_missing in enumerate(missing_representations):
                for i, (pt_id, _range) in enumerate(b_missing.items()):
                    _pt_map[bi][pt_id] = i
                    pooling_inp[bi, i, list(range(len(_range)))] = self._hidden_state_mem[l][bi, _range]
                    mask[bi, i, list(range(len(_range)))] = 1
            pooled_representations = self.pooler(pooling_inp, mask)[0]
            for bi in range(input_ids.shape[0]):
                for pt_id, mapping in _pt_map[bi].items():
                    self.pt_manager.pts[bi][pt_id].representation = pooled_representations[bi, mapping].unsqueeze(0)
        
        # ================================================================ #

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            self._hidden_state_mem[i] = torch.cat([self._hidden_state_mem[i], hidden_states], dim=1).to(hidden_states.device).to(torch.bfloat16)
            _update_pt_representations(i)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            self.pt_manager.flush_representations()

        hidden_states = self.norm(hidden_states)
        # self._hidden_state_mem = torch.cat([self._hidden_state_mem, hidden_states], dim=1).to(hidden_states.device)
        # _update_pt_representations()
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def set_req_gard_new_module(self, set_to=True):
        for p in self.pooler.parameters():
            p.requires_grad = set_to

    def set_req_grad_second_half(self, set_to=True):
        half_point = len(self.layers) // 2
        for _, l in enumerate(self.layers[half_point:]):
            for p in l.parameters():
                p.requires_grad = set_to
            l.self_attn.check_replace_pt_value()

@auto_docstring
class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, pt_ids_sorted):
        super().__init__(config)
        self.model = Qwen2Model(config, pt_ids_sorted)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._pt_ids_sorted = pt_ids_sorted
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)


        return PointerProbeOutput(
            loss=loss,
            logits=logits,

            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )



class Qwen2ForSequenceClassification(GenericForSequenceClassification, Qwen2PreTrainedModel):
    pass


class Qwen2ForTokenClassification(GenericForTokenClassification, Qwen2PreTrainedModel):
    pass


class Qwen2ForQuestionAnswering(GenericForQuestionAnswering, Qwen2PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen2PreTrainedModel",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "Qwen2RMSNorm",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2ForQuestionAnswering",
]
