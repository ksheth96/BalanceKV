import math
import torch
import transformers
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from flash_attn import flash_attn_func

from balanced_walk import balanced_walk, balanced_walk2


def manual_forward(
    model, 
    input_ids,
    kv_type,
    attention_mask=None,
    kv_cache=None, position_ids=None, cache_position=None, num_logits_to_keep=0,
    return_hidden_states=False,
):  
    hh = model.model.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(len(input_ids[0]), device=input_ids.device).unsqueeze(0)

    if int(transformers.__version__.split(".")[1]) >= 48:
        position_embeddings = model.model.rotary_emb(hh, position_ids)

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

        if int(transformers.__version__.split(".")[1]) >= 48:
            cos, sin = position_embeddings
        else:
            cos, sin = decoder_layer.self_attn.rotary_emb(vv, position_ids)
        qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
        d = qq.shape[-1]

        if q_len > 1:
            attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True, deterministic=False)      

        if kv_cache is None:
            if kv_type in ['exact2']:
                key_quant = kk
                val_quant = vv
            
            elif kv_type in ['weightedbw', 'weightedbw2', 'uniform']:
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

                if kv_type == 'weightedbw':
                    indices = balanced_walk(k_compressed, rng, gamma, temp, beta, itrs, block_size, value=v_compressed)
                elif kv_type == 'weightedbw2':
                    indices, weights = balanced_walk2(k_compressed, rng, gamma, temp, beta, itrs, block_size, value=v_compressed)
                elif kv_type == 'uniform':
                    indices = balanced_walk(k_compressed, rng, 0.0, temp, beta, itrs, block_size, value=v_compressed)
                
                k_bw = k_compressed.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,kk.shape[-1]))
                v_bw = v_compressed.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,vv.shape[-1]))
                if kv_type == 'weightedbw2':
                    weights = weights / 2**(itrs)
                    weights = weights.unsqueeze(-1)
                    v_bw = (v_bw * weights).to(vv.dtype)

                key_quant = torch.cat((kk[:,:,:sink_size], k_bw, kk[:, :, -window_size:]), dim=2)
                val_quant = torch.cat((vv[:,:,:sink_size], v_bw, vv[:, :, -window_size:]), dim=2)
                
                model.config.compress_size = k_bw.shape[2]
                # import pdb; pdb.set_trace()
            elif kv_type in ['sink']:
                sink_size = model.config.sink_size
                recent_size = model.config.recent_size
                key_quant = torch.cat((kk[:,:,:sink_size], kk[:, :, -recent_size:]), dim=2)
                val_quant = torch.cat((vv[:,:,:sink_size], vv[:, :, -recent_size:]), dim=2)
            else:
                import pdb; pdb.set_trace();
            past_kv_cache += [(key_quant, val_quant)]
        else:
            if kv_type in ['exact2']:
                assert len(kv_cache) == model.config.num_hidden_layers
                kk = torch.cat((kv_cache[i][0], kk), 2)
                vv = torch.cat((kv_cache[i][1], vv), 2)
                past_kv_cache += [(kk, vv)]
            
            elif kv_type in ['weightedbw', 'weightedbw2', 'uniform']:
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
            elif kv_type in ['sink']:
                key_old = kv_cache[i][0]
                val_old = kv_cache[i][1]
                if key_old.shape[1] != kk.shape[1]:
                    kk = repeat_kv(kk, key_old.shape[1]//kk.shape[1])
                    vv = repeat_kv(vv, val_old.shape[1]//vv.shape[1])
                
                kk = torch.cat((key_old, kk), dim=2)
                vv = torch.cat((val_old, vv), dim=2)

                attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True)
                past_kv_cache += [(kk, vv)]
            else:
                import pdb; pdb.set_trace();

            if kv_type in ['exact2']:
                attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True, deterministic=False)
        
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
    logits, cache, attention_mask = manual_forward(self, input_ids, position_ids=position_ids, num_logits_to_keep=1, kv_type=kv_type)
    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    for i in range(max_new_tokens - 1):
        position_ids = torch.tensor(input_ids.shape[-1] + i, device=input_ids.device).reshape(1,1)
        logits, cache, _ = manual_forward(self, pred_token_idx, position_ids=position_ids, attention_mask=attention_mask, kv_cache=cache, num_logits_to_keep=1, kv_type=kv_type)
        pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        # if generated_ids[-1] == 980:
        #     import pdb; pdb.set_trace();
        generated_ids.append(pred_token_idx.item())
        if pred_token_idx in eos_token_id:
            break
    sequences = torch.tensor(input_ids[0].tolist() + generated_ids, device=input_ids.device).unsqueeze(0)
    if not return_dict_in_generate:
        return sequences
    return GenerateDecoderOnlyOutput(sequences=sequences, past_key_values=cache)



if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, _attn_implementation='flash_attention_2')
    model = model.eval().requires_grad_(False)

    # input_ids = torch.randint(128000, (1, 128)).to(model.device)
    # out = model(input_ids=input_ids, num_logits_to_keep=1)

    # out2 = manual_forward(model, input_ids, kv_type='exact', num_logits_to_keep=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # if 'qwen' in model_name.lower():
    #     model.config.head_dim = 128
    #     model._supports_num_logits_to_keep = model._supports_logits_to_keep

    prompt_format = "Please complete the code given below. \n{context}Next line of code:\n"
    dataset = 'lcc'
    examples = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
    eg = examples[0]
    input_text = prompt_format.format(**eg)

    # prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    input_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    seq_len = input_ids.shape[-1]

    max_len = 64 #12
    terminators = [tokenizer.eos_token_id]
    kv_cache = None
    outputs = model.generate(input_ids, max_new_tokens=max_len, eos_token_id=terminators, do_sample=False, temperature=None, top_p=None, use_cache=True, past_key_values=kv_cache, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)
    print(outputs.sequences[:, seq_len:])
    # outputs.past_key_values[0][0][0, 0, seq_len]

    out = greedy_generate(model, input_ids, max_new_tokens=max_len, kv_type='exact2', eos_token_id=terminators, return_dict_in_generate=True)
    print(out.sequences[..., seq_len:])