import math
from typing import List, Optional, Tuple, Union, Any, Dict
import torch


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


def balanced_walk(key, rng, gamma_, temp_, beta_, itrs, block_size, value=None, sort_idx=None):
    if itrs == 0:
        return sort_idx
    b, h, n, d = key.shape

    if type(gamma_) != list:
        gamma_ = [gamma_] * itrs

    if type(block_size) != list:
        block_size = [block_size] * itrs

    for t in range(itrs):
        if sort_idx is not None:
            key_sorted, value_sorted = indexing(key, sort_idx, block_size[t], value)
            key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
            if value is not None:
                value_sorted = value_sorted.view(b, h, -1, block_size[t], d)
        else:
            new_n = math.ceil(n / block_size[t]) * block_size[t]
            key_sorted = torch.nn.functional.pad(key, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)
            if value is not None:
                value_sorted = torch.nn.functional.pad(value, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)

        normal_keys = key_sorted - torch.mean(key_sorted, dim=-2, keepdim=True)
        kernel_ = torch.exp(temp_ * torch.einsum('...nd,...sd->...ns', normal_keys, normal_keys)/math.sqrt(d) - beta_)

        if value is not None:
            kernel_ *= (1e-8 + torch.einsum('...nd,...sd->...ns', value_sorted, value_sorted))

        signs = torch.zeros(kernel_.shape[:4], dtype=torch.int16, device=kernel_.device)
        signs[:, :, :, 0] = 1
        partial_quad_form = kernel_[:, :, :, 0, 0].detach().clone()
        rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)
        for i in range(1, kernel_.shape[3]):
            partial_inner_prod = (kernel_[:, :, :, i, :i] * signs[:, :, :, :i]).sum(dim=-1)
            samp_prb = 0.5 - gamma_[t] * partial_inner_prod
            signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1
            partial_quad_form += (2 * signs[:, :, :, i] * partial_inner_prod + kernel_[:, :, :, i, i])

        signs = signs.view(b, h, -1)[:, :, :n]
        signs_argsort = torch.argsort(signs, dim=-1, stable=True)
        n = n//2
        if sort_idx is None:
            sort_idx = signs_argsort[:, :, :n]
        else:
            sort_idx = sort_idx.gather(2, signs_argsort[:, :, :n])
    return sort_idx
