from tqdm import tqdm
import time
import datetime
import logging
import os
import argparse
import json
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset

from utils import seed_everything, get_exp_name, set_logger, reset_logging, apply_template, compute_cache_size
from utils_longbench import dataset2prompt, dataset2maxlen, compute_score

from generate import greedy_generate

from minference import MInference
from minference.patch import minference_patch, new_patch
from minference.modules.kivi import KiviCache
from minference.modules.kvcompression import SnapKVCache, PyramidKVCache, StreamingLLMKVCache


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    # parser.add_argument("--output_dir", type=str, default="./result_longbench_v1", help="The directory of data.")
    parser.add_argument('--kv_type', type=str, default="exact")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--dataset', type=str, default='triviaqa')
    parser.add_argument('--debug', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--print_pred', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--opt_template', action='store_true', help="use optimized template")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")

    parser.add_argument("--group_size", type=int, default=32, help="for KIVI")
    parser.add_argument("--bits", type=int, default=2, choices=[1,2,3,4], help="for KIVI")
    parser.add_argument("--residual_length", type=int, default=32, help="for KIVI")

    parser.add_argument("--sink_size", type=int, default=32, help="for StreamingLLM")
    parser.add_argument("--window_size", type=int, default=32, help="for SnapKV,PyramidKV")
    parser.add_argument("--max_capacity_prompt", type=int, default=4096, help="for SnapKV,PyramidKV")
    parser.add_argument("--kernel_size", type=int, default=5, help="for SnapKV,PyramidKV")
    parser.add_argument("--pooling", type=str, default="avgpool", help="for SnapKV,PyramidKV")

    parser.add_argument("--headkv_method", type=str, default='adaptive', help="for HeadKV")
    parser.add_argument("--base_capacity", type=int, default=128, help="for HeadKV")
    parser.add_argument("--block_size", type=int, default=128, help="for HeadKV")
    parser.add_argument("--itrs", type=int, default=2, help="for BalancedWalk")
    parser.add_argument("--beta", type=float, default=0.0, help="for BalancedWalk")
    parser.add_argument("--temp", type=float, default=1.0, help="for BalancedWalk")
    parser.add_argument("--gamma", type=float, default=4.0, help="for BalancedWalk")
    parser.add_argument("--floor", type=float, default=0.2, help="for HeadKV")
    parser.add_argument("--head_choice", type=str, default='random', help="for HeadKV")

    parser.add_argument("--g", type=int, default=0, help="for Kernel Halving")

    return parser.parse_args(args)


def main():
    args = parse_args()
    seed_everything(args.seed)

    model_name = args.model_name
    real_model_name = model_name.split("/")[-1]
    exp_name = get_exp_name(real_model_name, args)
    
    logger, output_path = set_logger(args, args.dataset, exp_name, task="longbench_v1" + "_e" if args.e else "")
    logger.info(f"output_path: {output_path}")
    logger.info(f"exp_name: {exp_name}")

    for n_, v_ in args.__dict__.items():
        logger.info(f"{n_:<20} : {v_}")

    # 1. load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, _attn_implementation='flash_attention_2')
    model = model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(model, '_supports_num_logits_to_keep'):
        model.config.head_dim = 128
        model._supports_num_logits_to_keep = model._supports_logits_to_keep

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    terminators = [tokenizer.eos_token_id]
    if 'llama-3' in model_name.lower():   
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    elif 'glm' in model_name.lower():
        terminators.append(tokenizer.convert_tokens_to_ids("<|endoftext|>"))


    # 2. pacth the model
    if args.kv_type in ['exact']:
        pass
    elif args.kv_type in ['kivi', 'snapkv', 'pyramidkv', 'streamingllm']:
        kv_kwargs = {}
        if args.kv_type == 'kivi':
            kv_kwargs = {"group_size": args.group_size, "bits": args.bits, "residual_length": args.residual_length}
        elif args.kv_type in ['snapkv', 'pyramidkv']:
            kv_kwargs = {"window_size": args.window_size, "max_capacity_prompt": args.max_capacity_prompt, "kernel_size": args.kernel_size, "pooling": args.pooling}
        minference_config = MInference(
            'dense',
            'llama-3',
            kv_type=args.kv_type,
            attn_kwargs=kv_kwargs,
        )
        model = new_patch(model, minference_config.config)
    elif args.kv_type in ['weightedbw', 'weightedbw2', 'uniform']:
        logger.info("patching balanced walk KV cache ...")
        model.model.config.rng = torch.Generator('cuda').manual_seed(args.seed)
        model.model.config.gamma = args.gamma
        model.model.config.beta = 0.
        model.model.config.temp = args.temp
        model.model.config.block_size = args.block_size
        model.model.config.itrs = args.itrs #2 if args.kv_type in ['bw', 'weightedbw'] else 1
        if args.kv_type in ['weightedbw', 'weightedbw2', 'uniform']:
            model.model.config.sink_size = args.sink_size
            model.model.config.window_size = args.window_size
    elif args.kv_type in ['sink']:
        model.model.config.sink_size = args.sink_size
    elif args.kv_type in ['adapkvq', 'adapkvqsimple', 'adapkvqrndrot', 'adapkvqrndrot2', 'adapkvqrndrotsimple', 'adapkvqrndrot2simple']:
            model.model.config.window_size = args.window_size
            model.model.config.seed = args.seed
            if args.kv_type in ['adapkvqrndrot', 'adapkvqrndrot2', 'adapkvqrndrotsimple', 'adapkvqrndrot2simple']:
                rot_mat = torch.randn(model.model.config.head_dim, model.model.config.head_dim, generator=torch.Generator().manual_seed(args.seed))
                rot_mat = torch.linalg.qr(rot_mat)[0].unsqueeze(0).unsqueeze(0)
                model.model.config.rot_mat = rot_mat.to(device=model.device, dtype=model.dtype)
    else:
        raise NotImplementedError

    # 3. load dataset
    if args.e:
        examples = load_dataset('THUDM/LongBench', f"{args.dataset}_e", split='test')
    else:
        examples = load_dataset('THUDM/LongBench', f"{args.dataset}", split='test')

    prompt_format = dataset2prompt[args.dataset]
    maxlen = dataset2maxlen[args.dataset]

    if '32b' in model_name.lower() or '70b' in model_name.lower():
        max_input_length = 32_000
    else:
        max_input_length = 100_000

    # 4. predict
    tic0 = time.time()
    scores = []
    preds = []
    cnt = 0
    input_lens = []
    # if '32b' in model_name.lower() and args.dataset == 'gov_report':
    #     examples = [examples[234]]
    for eg in examples:
        tic = time.time()

        # 4.1 prepare input
        input_text = prompt_format.format(**eg)
        input_text = apply_template(model_name, tokenizer, args.dataset, input_text)
        input_tokens = tokenizer.encode(input_text)
        if len(input_tokens) > max_input_length:
            split = max_input_length // 2
            input_tokens = input_tokens[:split] + input_tokens[-split:]
        input_tensors = {"input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)}
        seq_len = len(input_tokens)

        # 4.2 set up KV cache type
        if args.kv_type == 'kivi':
            kv_cache = KiviCache(minference_config.config)
        elif args.kv_type == 'snapkv':
            minference_config.config.attn_kwargs['max_capacity_prompt'] = max(int(seq_len * 3.875 / 64), args.window_size+4)
            kv_cache = SnapKVCache(minference_config.config)
        elif args.kv_type == 'pyramidkv':
            minference_config.config.attn_kwargs['max_capacity_prompt'] = max(int(seq_len * 3.875 / 64), args.window_size+4)
            minference_config.config.num_layers = model.config.num_hidden_layers
            kv_cache = PyramidKVCache(minference_config.config)
            kv_cache.max_capacity_prompt = int(seq_len * 3.875 / 64)
        elif args.kv_type == 'headkv':
            kv_cache = DynamicCacheSplitHeadFlatten()
            model.model.config.base_capacity = int(seq_len * 3.875 / 64)
        elif args.kv_type == 'streamingllm':
            minference_config.config.attn_kwargs['n_init'] = max(int(seq_len * 3.875 / 64) - args.window_size, 4)
            minference_config.config.attn_kwargs['n_local'] = args.window_size
            kv_cache = StreamingLLMKVCache(minference_config.config)
        else:
            kv_cache = None
            if args.kv_type == 'sink':
                model.model.config.recent_size = seq_len//4 - args.window_size

        # 4.4 generate
        if args.kv_type in ['exact2', 'weightedbw', 'weightedbw2', 'sink', 'adapkvq', 'adapkvqsimple', 'adapkvqrndrot', 'adapkvqrndrot2', 'adapkvqrndrotsimple', 'adapkvqrndrot2simple']:
            outputs = greedy_generate(model, input_tensors['input_ids'], max_new_tokens=maxlen, kv_type=args.kv_type, eos_token_id=terminators, return_dict_in_generate=True)
        else:
            outputs = model.generate(**input_tensors, max_new_tokens=maxlen, eos_token_id=terminators, do_sample=False, temperature=None, top_p=None, use_cache=True, past_key_values=kv_cache, pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)
        
        if type(outputs) is not torch.Tensor:
            assert hasattr(outputs, "past_key_values")
            kv_cache_tensors = outputs.past_key_values
            outputs = outputs.sequences
        else:
            kv_cache_tensors = None
        
        kv_cache_size, kv_cache_shapes, kv_cache_size_ori = compute_cache_size(kv_cache_tensors, outputs.shape[-1], model_name, model.config)

        kv_cache_num = model.config.head_dim*(outputs.shape[-1]-1)*2*model.config.num_key_value_heads*model.config.num_hidden_layers
        kv_size_info = (f"kv_size: {kv_cache_size/1024**3:.2f} GB ({kv_cache_size_ori/1024**3:.2f} GB)" if kv_cache_size>0 else "")+ f", {kv_cache_size*8/(kv_cache_num):.3f}-bits"
        del kv_cache_tensors
        if cnt == 0:
            print(f"different kv_shape : {len(set(kv_cache_shapes))}, shape: {set(kv_cache_shapes)}")

        output = outputs[0, seq_len:]
        output_token_len = len(output)
        output = tokenizer.decode(output, skip_special_tokens=True)
        pred = output.strip()
        if args.dataset not in ['lcc', 'repobench-p']:
            pred = pred.lstrip('\n').split('\n')[0]
            pred = pred.split('  ')[0]

        ground_truths = eg['answers']
        if args.print_pred:
            print("=" * 200)
            print(f"pred:\n{pred}")
            print(f"label:\n{eg['answers'][0]}")
        score = compute_score(pred, ground_truths, eg['all_classes'], args.dataset)
        preds.append({"id": cnt, "prediction": pred, "ground_truth": ground_truths, "score": score})
        scores.append(score)
        input_lens.append(seq_len)

        toc = time.time()

        mem_str = f"mem: ({torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB), {kv_size_info}"

        logger.info(f"[{cnt}/{len(examples)}] score: {score:.3f} (avg: {torch.tensor(scores).mean():.3f}), slen: {seq_len} (avg: {np.mean(input_lens):.1f}), olen: {output_token_len}, time: ({toc-tic:.2f} sec, {toc-tic0:.2f} sec), {mem_str}")

        if not args.debug:
            with open(output_path, "w", encoding="utf8") as fout:
                for line in preds:
                    fout.write(json.dumps(line, ensure_ascii=False) + "\n")

        cnt += 1
        del outputs, kv_cache
        torch.cuda.empty_cache()

    logger.info("done.")
    reset_logging()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


    
