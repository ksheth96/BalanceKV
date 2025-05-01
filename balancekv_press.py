from typing import Optional, Tuple
import math
import sys
from dataclasses import dataclass

import torch
from torch import nn
from contextlib import contextmanager

from kvpress.presses.base_press import BasePress

from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def llamaattention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value = None,
    cache_position = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    q_len = query_states.shape[2]
    if q_len > 1:
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    else:
        assert past_key_value is not None
        itrs = past_key_value.itrs
        compress_size = past_key_value.compress_size
        log_weights = past_key_value.log_weights
        sink_size = past_key_value.sink_size
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        n_repeat = self.config.num_attention_heads // self.config.num_key_value_heads
        
        log_weights_used = log_weights
        log_weights_used = log_weights_used.repeat_interleave(n_repeat, dim=1).transpose(-1, -2)
        log_weights_used = log_weights_used.unsqueeze(0)

        qk = query_states @ repeat_kv(key_states, n_repeat).transpose(-1,-2) / self.config.head_dim**0.5
        bias = torch.zeros_like(qk)
        bias[:, :, :, sink_size:sink_size+compress_size] = log_weights_used
        attn_output = ((qk + bias).softmax(dim=-1) @ repeat_kv(value_states, n_repeat)).transpose(1,2)
        attn_weights = None

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def indexing(key, sort_idx, block_size, value=None):
    indices = sort_idx.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    new_n = math.ceil(sort_idx.shape[-1] / block_size) * block_size
    if new_n < sort_idx.shape[-1]:
        import pdb; pdb.set_trace();
    out_key = torch.nn.functional.pad(key.gather(2, indices), (0,0,0,new_n-sort_idx.shape[-1]), mode='constant', value=0.)
    out_value = None
    if value is not None:
        out_value = torch.nn.functional.pad(value.gather(2, indices), (0,0,0,new_n-sort_idx.shape[-1]), mode='constant', value=0.)
    return out_key, out_value


def balanced_walk(key, rng, gamma_, temp_, beta_, itrs, block_size, value=None, needle_mask = None, layer = None, sort_idx=None, query=None, qquery = None):
  b, h, n, d = key.shape
  if type(gamma_) != list:
      gamma_ = [gamma_] * itrs
  const_denom = 0.025 #change this to 0.00 to change the kernel back
  needle_mask_bw = None

  if type(block_size) != list:
      block_size = [block_size] * itrs
  weight_idx = None
  for t in range(itrs): #write range(1, itrs) to check everything still works
    if needle_mask_bw is not None:
        needle_mask_bw = torch.nn.functional.pad(needle_mask_bw, (0, math.ceil(n / block_size[t]) * block_size[t] - needle_mask_bw.shape[-1])).view(b, h, -1, block_size[t])
    if sort_idx is not None:
      key_sorted, value_sorted = indexing(key, sort_idx, block_size[t], value)
      key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
      if value is not None:
        weight_idx_padded = torch.nn.functional.pad(weight_idx, (0, math.ceil(n / block_size[t]) * block_size[t] - weight_idx.shape[-1]))
        value_sorted = value_sorted*weight_idx_padded.unsqueeze(-1)
        value_sorted = value_sorted.view(b, h, -1, block_size[t], d)
    else:
      new_n = math.ceil(n / block_size[t]) * block_size[t]
      key_sorted = torch.nn.functional.pad(key, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)
      value_sorted = None
      if value is not None:
        value_sorted = torch.nn.functional.pad(value, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)

    new_n = math.ceil(n / block_size[t]) * block_size[t]
    normal_keys = key_sorted - torch.mean(key_sorted, dim=-2, keepdim=True)
    normal_keys = normal_keys.view(b, h, new_n, d)
    if qquery is not None:
      normal_keys = qquery + normal_keys
    normal_keys = normal_keys.view(b, h, -1, block_size[t], d)

    if query is not None:
      query_key_correlation = torch.softmax(torch.einsum('b h n d,b h s m d->b h s n m',query[:,::4,:,:],normal_keys),dim=-1).mean(-2,keepdim=True)
      kernel_ = query_key_correlation*query_key_correlation.transpose(-1,-2)
    else:
      kernel_ = torch.exp(temp_ * torch.einsum('...nd,...sd->...ns', normal_keys, normal_keys)/math.sqrt(d) - beta_)
    if value is not None:

      kernel_ *= (1e-8 + torch.einsum('...nd,...sd->...ns', value_sorted, value_sorted)+const_denom)
    key_correlation = (1e-8 + torch.einsum('...nd,...sd->...ns', key_sorted, key_sorted))
    key_inner_prods = key_correlation.sum(dim = -1)

    if layer==1 and t==0:
        threshold = 0.0
        key_correlation = (1e-8 + torch.einsum('...nd,...sd->...ns', key_sorted, key_sorted))
        key_correlation_sum = key_correlation.sum(dim=-1)/key_correlation.shape[-1]
        needle_mask = key_correlation_sum > threshold
        needle_mask = needle_mask.view(b, h, -1)[:, :, :n]
        if n==0: #simply to deal with n==0, does not matter what we return, as long as the sha
            sort_idx = needle_mask[:, :, :0]
            weigth_idx = needle_mask[:, :, :0]
            needle_mask = needle_mask[:, :, :0]
            break
        needle_mask = needle_mask.to(torch.int32)
        needle_mask_padded = torch.nn.functional.pad(needle_mask, (25, 25), mode='constant', value=1)
        unfolded = needle_mask_padded.unfold(-1, 51, 1)
        result = unfolded.sum(dim=-1)
        #result = torch.round(result)
        #result_padded = torch.ones_like(needle_mask)
        #result_padded[:,  result
        new_needle_mask = result > 50
        needle_mask = new_needle_mask*needle_mask
        zero_counts = (needle_mask == 0).sum(dim=-1)  # Shape: (b, h, w)
        needle_mask = torch.nn.functional.pad(needle_mask, (0, math.ceil(n / block_size[t]) * block_size[t] - needle_mask.shape[-1])).view(b, h, -1, block_size[t])
        # #Count occurrences of 0 along the last dimension




    signs = torch.zeros(kernel_.shape[:4], dtype=torch.float32, device=kernel_.device)
    signs[:, :, :, 0] = 1
    rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)

    if needle_mask == None:
      needle_mask = torch.ones_like(signs)
    if needle_mask_bw == None:
        needle_mask_bw = needle_mask

    for i in range(1, kernel_.shape[3]):
      partial_inner_prod = (kernel_[:, :, :, i, :] * signs * needle_mask_bw).sum(dim=-1)
      prev_sign = signs[:,:,:,i-1]
      samp_prb = 0.5 - gamma_[t] * partial_inner_prod #+ 0.1*prev_sign


      signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1


    signs = signs.view(b, h, -1)[:, :, :n]
    needle_mask_bw = needle_mask_bw.view(b, h, -1)[:, :, :n]

    signs = signs*needle_mask_bw

    if signs.shape[-1]==0: #simply to deal with n==0
      sort_idx = signs[:, :, :0]
      weigth_idx = signs[:, :, :0]
      break
    if torch.all(needle_mask_bw):
        cumsum_neg = (signs == -1).cumsum(dim=-1)
        cumsum_pos = (signs == 1).cumsum(dim=-1)

        c_neg = torch.argmax((cumsum_neg == n//2).to(torch.int64), dim=-1) # Shape (b, h)
        c_pos = torch.argmax((cumsum_pos == n//2).to(torch.int64), dim=-1) # Shape (b, h)
        c = torch.maximum(c_neg, c_pos)

        # Ensure `c` is on the same device as `signs`
        c = c.to(signs.device)

        weight = signs

        # Create an index tensor `[0, 1, ..., n-1]` for comparison
        indices = torch.arange(signs.shape[2], device=signs.device).view(1, 1, -1)

        # 1Set all values after `c[a, b]` to `1`
        mask_after_c = indices > c.unsqueeze(-1)  # True for all d > c[a, b]

        weight[mask_after_c] = torch.abs(weight[mask_after_c])  # Set those indices to `1`

        # 2Identify where `signs[a, b, c[a, b]] == 1`
        mask_flip_needed = (signs.gather(2, c.unsqueeze(-1)) == 1).squeeze(-1)

        # Create mask for all indices `<= c[a, b]`
        mask_before_c = indices <= c.unsqueeze(-1)
        weight[mask_before_c] *= 2

        # Apply flipping only when `signs[a, b, c] == 1`
        flip_mask = mask_before_c & mask_flip_needed.unsqueeze(-1)
        weight[flip_mask] *= -1  # Flip selected values


        weight_argsort = torch.argsort(-weight, dim=-1, stable=True)
    else:
        weight = signs
        #flipping_mask = (signs == 1).sum(dim=-1) > (signs == -1).sum(dim=-1)
        #weight[flipping_mask] *= -1
        weight_zeros = weight == 0
        weight[weight_zeros] += 2
        weight += 1
        weight_argsort = torch.argsort(-weight, dim=-1, stable=True)
        weight[weight_zeros] = 1


    n = n//2
    if sort_idx is None:
      sort_idx = weight_argsort[:, :, :n]
      weight_idx = weight.gather(-1, weight_argsort[:, :, :n])
      needle_mask_bw = needle_mask_bw.gather(-1, weight_argsort[:, :, :n])
    else:
      sort_idx = sort_idx.gather(2, weight_argsort[:, :, :n])
      weigth_idx_1 = weight.gather(-1, weight_argsort[:, :, :n])
      weight_idx = weight_idx.gather(-1, weight_argsort[:, :, :n])
      weight_idx = weight_idx*weigth_idx_1
      needle_mask_bw = needle_mask_bw.gather(-1, weight_argsort[:, :, :n])


  return sort_idx, weight_idx, needle_mask


@dataclass
class BalanceKVPress(BasePress):
    """
    BalanceKV: 
    https://arxiv.org/abs/xxxx
    This method is a wrapper for any ScorerPress.
    """
    gamma: float = 4.0
    beta: float = 0.0
    block_size: int = 128
    window_size: int = 64
    sink_size: int = 16
    seed: int = 1234
    temp: float = 1.0
    rng : Optional[torch.Generator] = None
    itrs: int = 2

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.rng is None:
            self.rng = torch.Generator(device=keys.device)
            self.rng.manual_seed(self.seed)

        k_compressed = keys[:, :, self.sink_size:-self.window_size] 
        v_compressed = values[:, :, self.sink_size:-self.window_size]

        #indices, weights = balanced_walk(k_compressed, self.rng, self.gamma, self.temp, self.beta, self.itrs, self.block_size, value=v_compressed)

        if module.layer_idx==1: #detect the needle in the second layer - first layer does not contain it
            indices, weights, needle_mask = balanced_walk(k_compressed, self.rng, self.gamma, self.temp, self.beta, self.itrs, self.block_size, layer = module.layer_idx, value=v_compressed, qquery = qq_selected)
            #elif kv_type == 'uniform':
                #indices, weights, needle_mask = balanced_walk(k_compressed, self.rng, 0.0, self.temp, self.beta, self.itrs, self.block_size, layer = module.layer_idx, value=v_compressed, qquery = qq_selected)
        elif module.layer_idx > 1:
            indices, weights, _ = balanced_walk(k_compressed, self.rng, self.gamma, self.temp, self.beta, self.itrs, self.block_size, layer = module.layer_idx, needle_mask = needle_mask, value=v_compressed, qquery = qq_selected)
             #elif kv_type == 'uniform':
               # indices, weights, _ = balanced_walk(k_compressed, self.rng, 0.0, self.temp, self.beta, self.itrs, self.block_size, layer = module.layer_idx, needle_mask = needle_mask, value=v_compressed, qquery = qq_selected)
        else:
            indices, weights, _ = balanced_walk(k_compressed, self.rng, self.gamma, self.temp, self.beta, self.itrs, self.block_size, layer = module.layer_idx, value=v_compressed, qquery = qq_selected)
             #elif kv_type == 'uniform':
                #indices, weights, _ = balanced_walk(k_compressed, self.rng, 0.0, self.temp, self.beta, self.itrs, self.block_size, layer = module.layer_idx, value=v_compressed, qquery = qq_selected)
                 
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        k_bw = k_compressed.gather(2, indices).contiguous()
        v_bw = v_compressed.gather(2, indices).contiguous()

        if weights != None:#simply to deal with n==0
            weights_zeros = weights > 0
            weights_zeros = weights_zeros.unsqueeze(-1)
            v_bw_num = v_bw*weights_zeros
            v_bw_num = (v_bw_num).to(torch.bfloat16)
        else:
            v_bw_num = v_bw
        weights = weights.unsqueeze(-1)
        log_weights = torch.where(weights > 0, torch.log(weights), torch.full_like(weights, -1e9))

        keys = torch.cat((keys[:,:,:self.sink_size], k_bw, keys[:, :, -self.window_size:]), dim=2)
        values = torch.cat((values[:,:,:self.sink_size], v_bw_num, values[:, :, -self.window_size:]), dim=2)

        kwargs['past_key_value'].itrs = self.itrs
        kwargs['past_key_value'].compress_size = k_bw.shape[2]
        kwargs['past_key_value'].input_len = hidden_states.shape[1]
        kwargs['past_key_value'].sink_size = self.sink_size
        kwargs['past_key_value'].size = keys.shape
        kwargs['past_key_value'].log_weights = log_weights
        
        return keys, values

    
    @contextmanager
    def __call__(self, model):
        """
        Context manager to apply a compression method to a model.
        """

        hooks = []
        hooks_decoding = []
        try:
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
                layer.self_attn.forward = llamaattention_forward.__get__(layer.self_attn, layer.self_attn.__class__)
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
