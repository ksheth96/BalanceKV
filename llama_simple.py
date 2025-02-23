import math
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.generation.utils import GenerateDecoderOnlyOutput
from flash_attn import flash_attn_func
from balanced_walk import balanced_walk

def manual_forward_llama(
    model, 
    input_ids,
    kv_type,
    attention_mask=None,
    kv_cache=None, position_ids=None, cache_position=None, num_logits_to_keep=0,
):  
    hh = model.model.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(len(input_ids[0]), device=input_ids.device).unsqueeze(0)

    past_kv_cache = []
    for i, decoder_layer in enumerate(model.model.layers):
        # hh = decoder_layer(hh, position_ids=position_ids)[0]
        res = hh
        hh = decoder_layer.input_layernorm(hh)
    
        # h1, _, kv = decoder_layer.self_attn(hh, position_ids=position_ids, use_cache=False)
        # <===
        q_len = hh.shape[1]
        kv_len = q_len
        qq = decoder_layer.self_attn.q_proj(hh).reshape(1, q_len, -1, 128).transpose(1, 2)
        kk = decoder_layer.self_attn.k_proj(hh).reshape(1, kv_len, 8, 128).transpose(1, 2)
        vv = decoder_layer.self_attn.v_proj(hh).reshape(1, kv_len, 8, 128).transpose(1, 2)

        cos, sin = decoder_layer.self_attn.rotary_emb(vv, position_ids)
        qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
        d = qq.shape[-1]

        if q_len > 1:
            attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True)      

        if kv_cache is None:
            if kv_type in ['exact']:
                key_quant = kk
                val_quant = vv
            elif kv_type in ['weightedbw', 'uniform', 'bw', 'balancedwalk']:
                rng = model.config.rng
                gamma = model.config.gamma
                temp = model.config.temp
                beta = model.config.beta
                itrs = model.config.itrs
                block_size = model.config.block_size
                window_size = model.config.window_size
                sink_size = model.config.sink_size

                k_compressed = kk[:, :, sink_size:-window_size] 
                v_compressed = vv[:, :, sink_size:-window_size]

                if kv_type in ['weightedbw', 'bw', 'balancedwalk']:
                    indices = balanced_walk(k_compressed, rng, gamma, temp, beta, itrs, block_size, value=v_compressed)
                elif kv_type == 'uniform':
                    indices = balanced_walk(k_compressed, rng, 0.0, temp, beta, itrs, block_size, value=v_compressed)
                
                k_bw = k_compressed.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,kk.shape[-1]))
                v_bw = v_compressed.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,vv.shape[-1]))

                key_quant = torch.cat((kk[:,:,:sink_size], k_bw, kk[:, :, -window_size:]), dim=2)
                val_quant = torch.cat((vv[:,:,:sink_size], v_bw, vv[:, :, -window_size:]), dim=2)

                model.config.compress_size = k_bw.shape[2]
            else:
                import pdb; pdb.set_trace();
            past_kv_cache += [(key_quant, val_quant)]
        else:
            if kv_type in ['exact']:
                assert len(kv_cache) == model.config.num_hidden_layers
                kk = torch.cat((kv_cache[i][0], kk), 2)
                vv = torch.cat((kv_cache[i][1], vv), 2)
                past_kv_cache += [(kk, vv)]

            elif kv_type in ['weightedbw', 'bw', 'balancedwalk', 'uniform']:
                itrs = model.config.itrs
                compress_size = model.config.compress_size
                sink_size = model.config.sink_size

                key_old = kv_cache[i][0]
                val_old = kv_cache[i][1]
                kk = torch.cat((key_old, kk), dim=2)
                vv = torch.cat((val_old, vv), dim=2)

                qk = qq @ repeat_kv(kk, qq.shape[1]//kk.shape[1]).transpose(-1,-2) / d**0.5
                bias = torch.zeros_like(qk)
                bias[:, :, :, sink_size:sink_size+compress_size] = math.log(2**itrs)
                attn_output = ((qk + bias).softmax(dim=-1) @ repeat_kv(vv, qq.shape[1]//vv.shape[1])).transpose(1,2)
                past_kv_cache += [(kk, vv)]

            if kv_type in ['exact']:
                attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True)

        
        attn_output = attn_output.contiguous().view(qq.shape[0], qq.shape[2], -1)
        hh = decoder_layer.self_attn.o_proj(attn_output)
        # ===>
        hh = res + hh
    
        res = hh
        hh = decoder_layer.post_attention_layernorm(hh)
        hh = decoder_layer.mlp(hh)
        hh = res + hh

    hidden_states = model.model.norm(hh)  # (bsz, seq_len, dim)
    logits = model.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    return logits, past_kv_cache, attention_mask


@torch.no_grad()
def greedy_generate(self, input_ids, max_new_tokens, kv_type, eos_token_id=[128009], return_dict_in_generate=False):
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
    logits, cache, attention_mask = manual_forward_llama(self, input_ids, position_ids=position_ids, num_logits_to_keep=1, kv_type=kv_type)
    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    for i in range(max_new_tokens - 1):
        position_ids = torch.tensor(input_ids.shape[-1] + i, device=input_ids.device).reshape(1,1)
        logits, cache, _ = manual_forward_llama(self, pred_token_idx, position_ids=position_ids, attention_mask=attention_mask, kv_cache=cache, num_logits_to_keep=1, kv_type=kv_type)
        pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        if pred_token_idx in eos_token_id:
            break
    sequences = torch.tensor(input_ids[0].tolist() + generated_ids, device=input_ids.device).unsqueeze(0)
    if not return_dict_in_generate:
        return sequences
    return GenerateDecoderOnlyOutput(sequences=sequences, past_key_values=cache)