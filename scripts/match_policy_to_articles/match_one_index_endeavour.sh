#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=30
#SBATCH --partition=isi


source /home1/spangher/.bashrc
conda activate /home1/spangher/miniconda3/envs/retriv-py39

# Parse named arguments
city_name=""
policy_text_output_file=""
policy_and_news_article_output_file=""
match_policies_to_transcripts=false
match_policies_to_news_articles=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --city_name) city_name="$2"; shift ;;
        --policy_text_output_file) policy_text_output_file="$2"; shift ;;
        --policy_and_news_article_output_file) policy_and_news_article_output_file="$2"; shift ;;
        --match_policies_to_transcripts) match_policies_to_transcripts=true ;;
        --match_policies_to_news_articles) match_policies_to_news_articles=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo
echo '----------------------------------------------'
echo "City name: $city_name"
echo "Policy text output file: $policy_text_output_file"
echo "Policy and news article output file: $policy_and_news_article_output_file"
echo "Match policies to transcripts: $match_policies_to_transcripts"
echo "Match policies to news articles: $match_policies_to_news_articles"
echo '----------------------------------------------'
echo

python_cmd="python retriv_match_files.py"
python_cmd+="   --city_name $city_name"
python_cmd+="   --policy_text_output_file $policy_text_output_file"
python_cmd+="   --policy_and_news_article_output_file $policy_and_news_article_output_file"

if [[ "$match_policies_to_transcripts" == true ]]; then
    python_cmd+="   --match_policies_to_transcripts"
fi
if [[ "$match_policies_to_news_articles" == true ]]; then
    python_cmd+="   --match_policies_to_news_articles"
fi

# Execute the command
eval $python_cmd
#echo $python_cmd
