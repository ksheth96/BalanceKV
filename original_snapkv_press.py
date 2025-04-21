from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.snapkv_press import SnapKVPress
from transformers.models.llama.modeling_llama import repeat_kv


@dataclass
class OriginalSnapKVPress(SnapKVPress):
    """
    Original version of SnapKV
    """
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_repeat = module.config.num_attention_heads // module.config.num_key_value_heads
        n_kept = int(q_len * (1 - self.compression_ratio) / n_repeat)
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = repeat_kv(keys, n_repeat).gather(2, indices).contiguous()
        values = repeat_kv(values, n_repeat).gather(2, indices).contiguous()

        return keys, values


    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        # scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        # scores = scores.mean(2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores
