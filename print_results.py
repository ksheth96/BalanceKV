import os
import json
import numpy as np
from datasets import load_dataset

LONGBENCH_V1_E = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

LONGBENCH_V1 = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]


DATASET_SIZE = {'narrativeqa': 200, 'qasper': 200, 'multifieldqa_en': 150, 'multifieldqa_zh': 200, 'hotpotqa': 200, '2wikimqa': 200, 'musique': 200, 'dureader': 200, 'gov_report': 200, 'qmsum': 200, 'multi_news': 200, 'vcsum': 200, 'trec': 200, 'triviaqa': 200, 'samsum': 200, 'lsht': 200, 'passage_count': 200, 'passage_retrieval_en': 200, 'passage_retrieval_zh': 200, 'lcc': 500, 'repobench-p': 500}
DATASET_SIZE_E = {'qasper': 224, 'multifieldqa_en': 150, 'hotpotqa': 300, '2wikimqa': 300, 'gov_report': 300, 'multi_news': 294, 'trec': 300, 'triviaqa': 300, 'samsum': 300, 'passage_count': 300, 'passage_retrieval_en': 300, 'lcc': 300, 'repobench-p': 300}

CATEGORY2DATASET_E = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ['hotpotqa', '2wikimqa'],
    "Summarization": ['gov_report', 'multi_news'],
    "Few-shot Learning": ["trec", "triviaqa", "samsum"],
    "Synthetic Tasks": ["passage_count", "passage_retrieval_en"],
    "Code Completion": ["lcc", "repobench-p"]
}

CATEGORY2DATASET = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ['hotpotqa', '2wikimqa', 'musique', 'dureader'],
    "Summarization": ['gov_report', 'qmsum', 'multi_news', 'vcsum'],
    "Few-shot Learning": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic Tasks": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code Completion": ["lcc", "repobench-p"]
}

DATASET2CATEGORY = {
    "qasper": "Single-Doc QA",
    "multifieldqa_en": "Single-Doc QA",
    "hotpotqa": "Multi-Doc QA",
    "2wikimqa": "Multi-Doc QA",
    "gov_report": "Summarization",
    "multi_news": "Summarization",
    "trec": "Few-shot Learning",
    "triviaqa": "Few-shot Learning",
    "samsum": "Few-shot Learning",
    "passage_count": "Synthetic Tasks",
    "passage_retrieval_en": "Synthetic Tasks",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",
}

def print_longbench(model_name="Llama-3.1-8B-Instruct", result_path="./results", method_names=["exact", 'snapkv', 'snapkv2', 'pyramidkv', 'balancekv']):
    print(f"model_name: {model_name}")
    print(f"base_path: {result_path}")

    mname_map = {}
    for mname in os.listdir(os.path.join(result_path, "longbench-e")):
        mname_short = mname.split("_")[1]
        if mname_short in method_names:
            mname_map[mname_short] = mname

    print_first_line = False
    for method_name_short, method_name_full in mname_map.items():

        fnames = []
        base_path = os.path.join(result_path, "longbench-e", method_name_full)
        dataset_to_date = {}
        dataset_to_fn = {}
        for fn in os.listdir(base_path):
            dname = "_".join(fn.split("_")[:-1])
            date = fn.split("_")[-1].split(".")[0]
            if dname not in dataset_to_date:
                dataset_to_date[dname] = date
                dataset_to_fn[dname] = fn
            else:
                if int(date) > int(dataset_to_date[dname]):
                    dataset_to_date[dname] = date
                    dataset_to_fn[dname] = fn

        dataset_to_score = {}
        bpn_all = []
        for dname, fn in dataset_to_fn.items():
            scores = []
            for line in open(os.path.join(base_path, fn), "r"):
                json_line = json.loads(line)
                score = json_line['score']
                bpn_all.append(json_line['bpn'])
                scores.append(float(score))
            dataset_to_score[dname] = np.mean(scores)

        if not print_first_line:
            print(f"              | ", end='')
            # for dname in dataset_to_score.keys():
            for dataset in DATASET2CATEGORY.keys():
                if len(dataset) > 8:
                    print(f"{dataset[:4]+dataset[-4:]:<10}", end=' ')
                else:
                    print(f"{dataset:<10}", end=' ')
            print(" | average")
            print_first_line = True

        print(f"{method_name_short:<10}    | ", end='')
        scores_all = []
        # for dataset in dataset_to_score.keys():
        for dataset in DATASET2CATEGORY.keys():
            try:
                print(f"{dataset_to_score[dataset]:<10.4f}", end=' ')
                scores_all.append(dataset_to_score[dataset])
            except:
                print(f"{-2.:<10}", end=' ')
                scores_all.append(-np.inf)
        avg_score = np.mean(scores_all) if np.min(scores_all) > 0 else -1
        print(f" | {avg_score:<10.4f} ({method_name_short}, {np.mean(bpn_all):.2f}-bpn)")
    print("="*160)


def print_ruler(model_name = "Llama-3.1-8B-Instruct", result_path="./results", method_names=["exact", 'snapkv', 'snapkv2', 'balancekv'], datadir='8192'):
    print(f"model_name: {model_name}")
    print(f"sequence length: {datadir}")
    print(f"base_path: {result_path}")

    mname_map = {}
    for mname in os.listdir(os.path.join(result_path, "ruler")):
        mname_short = mname.split("_")[1]
        if mname_short in method_names:
            mname_map[mname_short] = mname
    
    print_first_line = False
    for method_name_short, method_name_full in mname_map.items(): 
        fnames = []
        date = 0
        base_path = os.path.join(result_path, "ruler", method_name_full)
        for fn in os.listdir(base_path):
            if datadir not in fn:
                continue
            if date < int(fn.split("_")[-1].split(".")[0]):
                date = int(fn.split("_")[-1].split(".")[0])
                fn_ = fn
        for line in open(os.path.join(base_path, fn_), "r"):
            res = json.loads(line)
        
        if not print_first_line:
            print("           ", end='|')
            for tn in res.keys():
                if len(tn) < 8:
                    print(f"{tn:>8}", end=' ')
                else:
                    print(f"{tn:>16}", end=' ')
            print(" | average")
            print_first_line = True

        print(f"{method_name_short:<10} ", end='|')
        values = []
        for tn, val in res.items():
            vv = list(val.values())[0]
            if len(tn) < 8:
                print(f" {vv:>7.2f} ", end='')
            else:
                print(f" {vv:>15.2f} ", end='')
            values.append(vv)
        print(f" | {np.mean(values):>7.2f} ")
    print("="*160)
    
        
def main():
    print("dataset: ruler")
    method_names = ["exact", 'snapkv', 'snapkv2', 'balancekv', 'pyramidkv']
    # print_ruler(result_path="./results", method_names=method_names, datadir='4096')
    # print_ruler(result_path="./results", method_names=method_names, datadir='8192')
    print_longbench(result_path="./results", method_names=method_names)
    

if __name__ == "__main__":
    main()