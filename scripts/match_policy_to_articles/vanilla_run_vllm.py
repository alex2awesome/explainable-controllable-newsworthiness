from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os
HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
proj_dir = '/project/jonmay_231/spangher/Projects/conditional-information-retrieval'
config_data = json.load(open(f'{proj_dir}/config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def load_model(model_name: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME, # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True
    )
    return tokenizer, model


def write_to_file(fname, ids, outputs):
    with open(fname, 'wb') as file:
        for _id, output in zip(ids, outputs):
            response = output.outputs[0].text
            response = unicodedata.normalize('NFKC', response)
            if response and url:
                output = {}
                output['id'] = str(_id)
                output['response'] = str(response)
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--id_col', type=str, default='article_url')
    parser.add_argument('--prompt_col', type=str, default='prompt')
    parser.add_argument('--output_file', type=str, default='llm_annotations.txt')
    parser.add_argument('--batch_size', type=str, default=100)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    prompt_df = pd.read_csv(args.prompt_file)
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(prompt_df)

    prompt_batches = list(batch(prompt_df, n=args.batch_size))

    tokenizer, model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    # generate the summaries
    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE
    for batch_df in tqdm(prompt_batches):
        fname, fext = os.path.splitext(args.output_file)
        output_fname = f'{fname}__info__{start_idx}_{end_idx}{fext}'
        if not os.path.exists(output_fname):
            # generate the informational summaries
            outputs = model.generate(batch_df[args.prompt_col].tolist(), sampling_params)
            write_to_file(output_fname, batch_df[args.id_col].tolist(), outputs)
        # update the indices
        start_idx = end_idx
        end_idx = start_idx + BATCH_SIZE





