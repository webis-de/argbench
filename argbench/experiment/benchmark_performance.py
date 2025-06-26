import sys
import pandas as pd
import gc
import torch
import seaborn as sns

from datasets import load_from_disk
from torch.utils.data import DataLoader
from time import perf_counter
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import SamplingParams, LLM
from vllm.distributed import destroy_model_parallel


dataset =load_from_disk("/bigwork/nhwpajjy/task-specific-argument-mining-and-generation-data/argbench-dataset/argbench-in-task-small")
generation_dataset = dataset["test_counter_argument_generation_cmv_hua18"]

longest_data_point=None
length = 0
all_truncated = {}

for data_point in DataLoader(generation_dataset,batch_size=1):
    if len(data_point["input"][0])> length:
        longest_data_point = data_point
        length = len(data_point["input"][0])

def benchmark_cutoffs(llm, tokenizer, min_tokens, max_tokens, step, model):
    print(f"benchmarking {model}")
    for cutoff_len in range(min_tokens, max_tokens, step):
        tokenized = tokenizer(longest_data_point["input"], max_length=cutoff_len, truncation=True, padding=True, return_tensors="pt")
        concatenated = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
        all_truncated[cutoff_len]= concatenated
    all_time = []
    all_cutoffs = []
    sampling_params={"temperature": 0.8, "top_p": 1, "top_k": -1, "max_tokens": 1000, "min_tokens": 999, "truncate_prompt_tokens":2}
    sampling_params = SamplingParams(**sampling_params )

    for cutoff in tqdm(all_truncated):
        t = perf_counter()
        llm.generate(all_truncated[cutoff], sampling_params=sampling_params)
        e = perf_counter()
        time = e - t
        all_time.append(time)
        all_cutoffs.append(cutoff)
    df_cutoffs = pd.DataFrame({ "cutoffs":all_cutoffs, "time":all_time})
    df_cutoffs["model"] = model

    return df_cutoffs

def benchmark_tokens(llm, min_tokens, max_tokens, step, model):
    all_params = {}
    for tokens in range(min_tokens, max_tokens, step):
        sampling_params={"temperature": 0.8, "top_p": 1, "top_k": -1, "max_tokens": tokens, "min_tokens": tokens-1, "truncate_prompt_tokens":2}
        sampling_params = SamplingParams(**sampling_params )
        all_params[tokens] = sampling_params
    all_time = []
    all_tokens = []
    for tokens in tqdm(all_params):
        sampling_params = all_params[tokens]
        t = perf_counter()
        llm.generate(longest_data_point["input"][0], sampling_params=sampling_params)
        e = perf_counter()
        time = e - t
        all_time.append(time)
        all_tokens.append(tokens)
    df_tokens = pd.DataFrame({ "tokens":all_tokens, "time":all_time})
    df_tokens["model"]=model
    return df_tokens

all_df_cutoffs = []
all_df_tokens = []
for model in tqdm(["DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "Mistral-7B-Instruct-v0.3", "Mistral-Nemo-Instruct-2407", "Mistral-Small-Instruct-2409"]):
    path = f"/bigwork/nhwpajjy/pre-trained-models/{model}"
    llm = LLM(model=path)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token_id = 0
    df_tokens = benchmark_tokens(llm, 50, 4000, 2000, model)
    df_cutoffs =  benchmark_cutoffs(llm, tokenizer, 50, 4000, 2000, model)
    all_df_cutoffs.append(df_cutoffs)
    all_df_tokens.append(df_tokens)
    del llm
    destroy_model_parallel()
    torch.cuda.empty_cache()
    gc.collect()

df_cutoffs = pd.concat(all_df_cutoffs)
df_cutoffs.to_csv("cutoffs.csv")
plot = sns.lineplot(df_cutoffs, x="cutoffs", y="time", hue="model")
fig = plot.get_figure()
fig.savefig("cutoffs.png")

df_tokens = pd.concat(all_df_tokens)
df_tokens.to_csv("tokens.csv")
plot = sns.lineplot(df_tokens, x="tokens", y="time", hue="model")
fig = plot.get_figure()
fig.savefig("tokens.png")