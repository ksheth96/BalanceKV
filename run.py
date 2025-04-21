# modified from https://github.com/NVIDIA/kvpress/blob/main/evaluation/evaluate.py
import json
import time
from typing import Any, Optional, Tuple, Dict
import random
import argparse
import numpy as np
import torch
from datasets import load_dataset

from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress import BasePress, RandomPress, SnapKVPress, PyramidKVPress
from balancekv_press import BalanceKVPress
from original_snapkv_press import OriginalSnapKVPress
from utils import set_logger, get_method_name, reset_logger

from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.models.llama.modeling_llama import repeat_kv

from evaluation.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from evaluation.longbench.calculate_metrics import dataset2metric
from evaluation.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from evaluation.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from evaluation.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from evaluation.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from evaluation.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from evaluation.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer


DATASET_DICT = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench-e",
    "longbench-v2": "Xnhyacinth/LongBench-v2",
}

LONGBENCH = ['2wikimqa', '2wikimqa_e', 'gov_report', 'gov_report_e', 'hotpotqa', 'hotpotqa_e', 'lcc_e', 'multi_news', 'multi_news_e', 'multifieldqa_en', 'multifieldqa_en_e', 'passage_count_e', 'passage_retrieval_en_e', 'qasper', 'qasper_e', 'repobench-p_e', 'samsum_e', 'trec_e', 'triviaqa_e']


SCORER_DICT = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer,
    "longbench-v2": longbenchv2_scorer,
}


PRESS_DICT = {
    "exact": RandomPress(),
    "random": RandomPress(),
    "snapkv": OriginalSnapKVPress(),
    "snapkv2": SnapKVPress(),
    "balancekv": BalanceKVPress(),
    "pyramidkv": PyramidKVPress(),
}

class DynamicCacheForGQA(DynamicCache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                # if n_heads is different, we need to repeat the cache
                n_repeat = self.key_cache[layer_idx].shape[1] // key_states.shape[1]
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], repeat_kv(key_states, n_repeat)], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], repeat_kv(value_states, n_repeat)], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class CustomKVPressTextGenerationPipeline(KVPressTextGenerationPipeline):
    def _forward(self, input_tensors, max_new_tokens, press, cache,):
        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]
        self.context_length = context_length
        # Continue with the original behavior
        return super()._forward(input_tensors, max_new_tokens, press, cache)


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=CustomKVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--dataset', type=str, default='longbench-e')
    parser.add_argument('--datadir', type=str, default='qasper_e')
    parser.add_argument('--method', type=str, default="snapkv")
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument("--compression_ratio", type=float, default=0.0)
    parser.add_argument("--window_size", type=int, default=32, help="for SnapKV,PyramidKV")
    parser.add_argument("--kernel_size", type=int, default=5, help="for SnapKV,PyramidKV")
    parser.add_argument("--block_size", type=int, default=128, help="for BalancedKV")
    parser.add_argument("--sink_size", type=int, default=32, help="for BalancedKV")
    parser.add_argument("--itrs", type=int, default=2, help="for BalancedKV")
    parser.add_argument("--beta", type=float, default=0.0, help="for BalancedKV")
    parser.add_argument("--temp", type=float, default=1.0, help="for BalancedKV")
    parser.add_argument("--gamma", type=float, default=4.0, help="for BalancedKV")
    return parser.parse_args(args)


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.compression_ratio == 0.0 and args.method != 'balancekv':
        args.method = 'exact'

    method_name = get_method_name(args)
    dataset_name = args.dataset + "-" + args.datadir.replace("_e", "")
    logger, save_filename = set_logger(args, dataset_name, method_name)

    for n_, v_ in args.__dict__.items():
        logger.info(f"{n_:<20} : {v_}")

    df = load_dataset(DATASET_DICT[args.dataset], args.datadir, split="test").to_pandas()
    if args.fraction < 1.0:
        df = df.sample(frac=args.fraction, random_state=args.seed)
    df["predicted_answer"] = None
    if args.dataset in ["longbench-e", "longbench"]:
        df_context = df.groupby("_id")
    else:
        df_context = df.groupby("context")
    scorer = SCORER_DICT[args.dataset]
    
    model_kwargs = {"torch_dtype": "auto"}
    model_kwargs["attn_implementation"] = "flash_attention_2"
    pipe = pipeline("kv-press-text-generation", model=args.model, device_map="auto", model_kwargs=model_kwargs)

    press = PRESS_DICT[args.method]
    if args.method == 'exact':
        pass
    elif args.method in ['snapkv', 'snapkv2']:
        press.compression_ratio = args.compression_ratio
        press.window_size = args.window_size
        press.kernel_size = args.kernel_size
    elif args.method == 'balancekv':
        press.window_size = args.window_size
        press.sink_size = args.window_size
        press.block_size = args.block_size
        press.itrs = args.itrs
        press.beta = args.beta
        press.temp = args.temp
        press.gamma = args.gamma
        press.seed = args.seed
    else:
        import pdb; pdb.set_trace();

    tic0 = time.time()
    max_context_length = None
    max_new_tokens = None
    metric_values = []
    cnt = 0
    n_data = len(df["context"])
    for context, df_ in df_context:
        if args.dataset in ["longbench-e", "longbench"]:
            context = df_["context"].values[0]
        tic = time.time()
        questions = df_["question"].to_list()
        max_new_tokens_ = max_new_tokens if max_new_tokens is not None else df_["max_new_tokens"].iloc[0]
        answer_prefix = df_["answer_prefix"].iloc[0]
        task_name = df_["task"].values[0]

        cache = DynamicCacheForGQA()
        output = pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens_,
            max_context_length=max_context_length,
            cache=cache
        )
        df.loc[df_.index, "predicted_answer"] = output["answers"]
        # df.loc[df_.index, "compression_ratio"] = press.compression_ratio  # type:ignore[attr-defined]

        input_len = pipe.context_length #cache.get_seq_length()
        cache_size = 0
        cache_shapes = []
        for k_cache in cache.key_cache:
            cache_size += k_cache.numel() * k_cache.element_size()
            cache_shapes.append(k_cache.shape)
        for v_cache in cache.value_cache:
            cache_size += v_cache.numel() * v_cache.element_size()
            cache_shapes.append(v_cache.shape)
        full_cache_numel = 2 * input_len * pipe.model.config.head_dim * pipe.model.config.num_key_value_heads * pipe.model.config.num_hidden_layers 
        full_cache_size = full_cache_numel * pipe.model.dtype.itemsize #/ 1024**3

        if cnt == 0:
            cache_shape = list(set(cache_shapes))[0]
            logger.info(f"kv_cache_shape: {cache_shape}")

        del cache
        torch.cuda.empty_cache()

        if args.dataset in ["longbench-e", "longbench"]:
            metric_value = scorer(df.loc[df_.index])
            metric_name = dataset2metric[args.datadir.replace("_e","")].__name__.replace("_score", "")
        else:
            metric_ = scorer(df.loc[df_.index])
            metric_name = list(metric_[task_name].keys())[0]
            metric_value = metric_[task_name][metric_name]
        
        metric_values.append(metric_value)

        toc = time.time()
        logger.info(
            f"{cnt}/{n_data} (id: {df_.index.values[0]:>4}), {metric_name}: {metric_value:.2f} (avg: {np.mean(metric_values):.2f})"+\
            f", task: {task_name}, slen: {input_len}, cache_size: {cache_size/2**30:.3f} GB ({full_cache_size/2**30:.3f} GB, {8*cache_size/full_cache_numel:.3f}-bits)"+\
            f", time: ({toc-tic:.2f} s, {toc-tic0:.2f} s)"
            # f", pred: {output['answers']}, ans: {df_['answer'].values[0]}"
        )
        cnt += 1

    # Calculate metrics
    metrics = scorer(df)
    with open(str(save_filename).replace(".csv", ".json"), "w") as f:
        json.dump(metrics, f)
    # print(f"Average compression ratio: {df['compression_ratio'].mean():.2f}")
    print(metrics)

    if not args.debug:
        reset_logger()


if __name__ == "__main__":
    main()