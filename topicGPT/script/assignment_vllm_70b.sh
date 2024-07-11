#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=isi

module load python/3.11
pip install -r requirements.txt
python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-70B-Instruct --dtype auto --api-key token-abc123
python setup.py --num_samples 2000
python script/assignment.py --deployment_name gpt-4 \
                        --max_tokens 300 --temperature 0.0 --top_p 0.0 \
                        --data data/input/sf_text_and_label_sample.jsonl \
                        --prompt_file prompt/sf_policies/sf_policies_assignment.txt \
                        --topic_file data/output/sf_policies/sf_policies_result_1.md \
                        --out_file data/output/sf_policies/sf_level1_500_assignment.jsonl \
                        --verbose True
python logistic_regression.py