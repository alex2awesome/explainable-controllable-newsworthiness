#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=15
#SBATCH --partition=isi

eval "$(conda shell.bash hook)"
conda activate py310

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ ! -d "$INPUT_DIR" ]; then
  gsutil cp gs://usc-data/newsworthiness-city-council-videos/$INPUT_DIR.tar.gz $INPUT_DIR.tar.gz
  tar -xvzf $INPUT_DIR.tar.gz
fi

#python $SCRIPT_DIR/transcribe_and_diarize.py \
python transcribe_and_diarize.py \
  --input_dir $1 \
  --output_dir $2

