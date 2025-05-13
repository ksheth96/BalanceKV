import torch
import numpy as np
import random
import os
import datetime
import logging
from pathlib import Path

import transformers
from minference.modules.kivi import KiviCache
from minference.modules.kvcompression import SnapKVCache, PyramidKVCache, StreamingLLMKVCache
DynamicCacheSplitHeadFlatten=None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_exp_name(model_name, args):
    exp_name = f"{model_name}_{args.kv_type}"
    if args.kv_type == 'kivi':
        exp_name += f"_g{args.group_size}_b{args.bits}_r{args.residual_length}"
    elif args.kv_type in ['snapkv', 'pyramidkv']:
        exp_name += f"_w{args.window_size}_nlinear_k{args.kernel_size}_p{args.pooling}"
    elif args.kv_type in ['headkv']:
        args.kernel_size = 7
        args.pooling = 'maxpool'
        args.beta = 1.5
        args.temp = 1.0
        exp_name += f"_{args.headkv_method}_w{args.window_size}_nlinear_k{args.kernel_size}_p{args.pooling}_b{args.beta}_t{args.temp}"
    elif args.kv_type in ['bw', 'balanced_walk', 'twostagebw', 'bw2']:
        #  best_param: (0.1, 0.2, 128)
        exp_name += f"_g{args.gamma}_t{args.temp}_b{args.block_size}_r{args.window_size}"
    elif args.kv_type in ['weightedbw', 'weightedbw2']:
        exp_name += f"_itr{args.itrs}_g{args.gamma}_t{args.temp}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['weightedbwlayer', 'weightedbwlayer1', 'weightedbwlayer2', 'weightedbwlayer3']:
        exp_name += f"_itr{args.itrs}_g{args.gamma}_t{args.temp}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['uniform']:
        exp_name += f"_itr{args.itrs}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['adapkvq', 'adapkvqsimple', 'adapkvqrndrot', 'adapkvqrndrot2', 'adapkvqrndrotsimple', 'adapkvqrndrot2simple']:
        exp_name += f"_s{args.seed}"
    elif args.kv_type in ['streamingllm', 'sink']:
        exp_name += f"_w{args.window_size}"
    elif args.kv_type in ['integer', 'integerchannel', 'integertoken']:
        exp_name += f"_b{args.bits}"
    elif args.kv_type in ['kh']:
        exp_name += f"_sink{args.sink_size}_g{args.g}"
    # print(f"exp_name: {exp_name}")
    return exp_name


def set_logger(args, dataset, exp_name, task="longbench_v1"):
    # 0. set up logging
    datestr = datetime.datetime.now().strftime('%y%m%d%H%M')
    task_path = f"./{args.prefix}results/{task}"
    log_path = f"./{args.prefix}logs/{task}"
    if not args.debug:
        result_dir = Path(os.path.join(task_path, exp_name))
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{dataset}_{datestr}.jsonl"
        log_path = os.path.join(log_path, exp_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = os.path.join(log_path, f"{dataset}_{datestr}.txt")
        # Configure the logger
        logging.basicConfig(
            level=logging.INFO, 
            format=f'[%(asctime)s]{dataset}|{args.kv_type}| %(message)s',
            datefmt='%y%m%d %H:%M:%S',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(
            level=logging.INFO, 
            format=f'[%(asctime)s]{dataset}|{args.kv_type}| %(message)s',
            datefmt='%y%m%d %H:%M:%S',
            handlers=[logging.StreamHandler()],
        )
        output_path = ""

    logger = logging.getLogger(__name__)
    return logger, output_path


def reset_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def apply_template(model_name, tokenizer, dataset, input_text):
    if 'llama-3' in model_name.lower() or 'llama-8b' in model_name.lower():
        # if dataset in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]:  
        #     msgs = [dict(role="system", content=input_text)]
        #     input_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        # input_text = f"[INST] {input_text} [/INST]"
        pass

    elif 'mistral' in model_name.lower() or 'ministral' in model_name.lower():
        # msgs = [{"role": "user", "content": input_text},]
        # input_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        pass
    elif 'glm' in model_name.lower():
        input_text = tokenizer.apply_chat_template([{"role": "user", "content": input_text}],
            add_generation_prompt=True,
            tokenize=False)
    elif 'gemma' in model_name.lower():
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text}
                ]
            }
        ]
        inputs = tokenizer.apply_chat_template(messages,  add_generation_prompt=True, tokenize=True, return_dict=True)
        input_text = inputs['input_ids']
    elif 'qwen/qwen2.5' in model_name.lower():
        if dataset not in ["gov_report", "multi_news", "trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # return_dict=True
            )
            # input_text = inputs['input_ids']
    elif 'qwen/qwq' in model_name.lower():
        if dataset not in ["gov_report", "multi_news", "trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
            messages = [
                {"role": "user", "content": input_text}
            ]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        

    # elif 'glm' in model_name.lower() or 'mistral' in model_name.lower():
    #     input_tokens = tokenizer(input_text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    else:
        import pdb; pdb.set_trace();
    return input_text


def compute_cache_size(kv_cache_tensors, output_len, model_name, model_config):
    # calculate the exact kv_cache size
    kv_cache_size = 0
    kv_cache_shapes = []
    if kv_cache_tensors is not None:
        if type(kv_cache_tensors) == tuple:
            for kv_per_layer in kv_cache_tensors:
                for kv in kv_per_layer:
                    kv_cache_size += kv.numel() * kv.element_size()
                    kv_cache_shapes.append(kv.shape)
            # kv_cache_size == 2 (k,v) * 2 (bytes) * 8 (nheads) * 128 (headim) * seq_len * 32 (num_layers) / 1024**3 if kv_type == 'exact'
        elif type(kv_cache_tensors) in [SnapKVCache, PyramidKVCache, DynamicCacheSplitHeadFlatten, StreamingLLMKVCache, transformers.cache_utils.HybridCache, transformers.cache_utils.DynamicCache]:
            for k_cache in kv_cache_tensors.key_cache:
                kv_cache_size += k_cache.numel() * k_cache.element_size()
                kv_cache_shapes.append(k_cache.shape)
            for v_cache in kv_cache_tensors.value_cache:
                kv_cache_size += v_cache.numel() * v_cache.element_size()
                kv_cache_shapes.append(v_cache.shape)
            # kv_cache_size == 2 * 2 * 32 * 128 * int(seq_len * 3.875 / 64) *32 / 1024**3
            # kv_cache_size += sum([k_cache.numel() * k_cache.element_size() for k_cache in kv_cache_tensors.key_cache]) + sum([v_cache.numel() * v_cache.element_size() for v_cache in kv_cache_tensors.value_cache])
            # aa = 2 * 2 * 32 * 128 * int(seq_len * 3.875 / 64) * 32  / 1024**3
        elif type(kv_cache_tensors) in [KiviCache]:
            for kvc in kv_cache_tensors.kv_cache:
                for cch in kvc:
                    kv_cache_size += cch.numel() * cch.element_size()
        elif type(kv_cache_tensors) == list:
            for kv_per_layer in kv_cache_tensors:
                for kv in kv_per_layer:
                    for kkvv in kv:
                        if kkvv is not None:
                            kv_cache_size += kkvv.numel() * kkvv.element_size()
                            kv_cache_shapes.append(kkvv.shape)
                        else:
                            import pdb; pdb.set_trace();
        else:
            import pdb; pdb.set_trace();
    else:
        import pdb; pdb.set_trace();

    # estimated kv_cache size
    if 'llama' in model_name.lower() or 'mistral' in model_name.lower() or 'gemma' in model_name.lower():
        kv_cache_size_ori = 2 * (output_len-1) * model_config.num_hidden_layers * model_config.head_dim * model_config.num_key_value_heads * 2
    elif 'glm' in model_name.lower():
        kv_cache_size_ori = 2 * (output_len-1) * model_config.num_hidden_layers * model_config.multi_query_group_num * model_config.kv_channels * 2
    elif 'qwen' in model_name.lower():
        kv_cache_size_ori = 2 * (output_len-1) * model_config.num_hidden_layers * (model_config.hidden_size / model_config.num_attention_heads) * 2
        # model.config.head_dim = 128
    else:
        import pdb; pdb.set_trace();

    return kv_cache_size, kv_cache_shapes, kv_cache_size_ori
