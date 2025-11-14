#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=150G
#SBATCH --partition=isi


source /home1/spangher/.bashrc
conda activate vllm-py310

start_idx=$1
step=$2
iterations=$3
iterations=$((iterations + 1))
end_idx=$((start_idx + step))

for ((i=0; i<iterations; i++)); do
    python vanilla_run_vllm.py \
      --start_idx ${start_idx} \
      --end_idx ${end_idx} \
      --prompt_file prompt_df.csv \
      --output_file llm_annotations.txt
      # --source_data_file sources_in_articles.jsonl

    start_idx=${end_idx}
    end_idx=$((start_idx + step))
done