#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

source /home1/spangher/.bashrc
conda activate retriv-py39


# Parse named arguments
index_name=""
embedding_model=Salesforce/SFR-Embedding-2_R
batch_size=1
max_seq_length=800
start_index=0
end_index=-1
filter_by_keywords=false
text_col=article_text
id_col=article_url
file_to_index=false
file_pattern_to_index=false
file_list=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --index_name) index_name="$2"; shift ;;     # Shift twice to get past the value
        --embedding_model) embedding_model="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --max_seq_length) max_seq_length="$2"; shift ;;
        --start_index) start_index="$2"; shift ;;
        --end_index) end_index="$2"; shift ;;
        --text_col) text_col="$2"; shift ;;
        --id_col) id_col="$2"; shift ;;
        --file_to_index) file_to_index="$2"; shift ;;
        --file_pattern_to_index) file_pattern_to_index="$2"; shift ;;
        --filter_by_keywords) filter_by_keywords=true ;;         # Set the flag to true if present
        --files_to_index)
            shift                                    # Move past the flag itself
            file_list=""                             # Initialize the file list as a string
            while [[ "$#" -gt 0 ]] && [[ "$1" != --* ]]; do   # Gather filenames until another flag
                file_list+="$1 "                    # Append the filename followed by a space
                shift
            done
            file_list="${file_list% }"              # Remove the trailing space
            continue
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;  # Handle unknown params
    esac
    shift   # Move to the next parameter
done

echo
echo '----------------------------------------------'
echo "Indexing file: $file_to_index"
echo "Indexing file pattern: $file_pattern_to_index"
echo "Indexing files: $file_list"
echo "Index name: $index_name"
echo "ID column: $id_col"
echo "Text column: $text_col"
echo "Embedding model: $embedding_model"
echo "Batch size: $batch_size"
echo "Max sequence length: $max_seq_length"
echo '----------------------------------------------'
echo

python_cmd="python retriv_index_files.py"
python_cmd+="     --index_name $index_name"
python_cmd+="   --embedding_model $embedding_model"
python_cmd+="   --batch_size $batch_size"
python_cmd+="   --max_seq_length $max_seq_length"
python_cmd+="   --start_index $start_index"
python_cmd+="   --end_index $end_index"
python_cmd+="   --text_col $text_col"
python_cmd+="   --id_col $id_col"

if [[ "$file_to_index" != false ]]; then
    python_cmd+="   --file_to_index $file_to_index"
fi
if [[ "$file_pattern_to_index" != false ]]; then
    python_cmd+="   --file_pattern_to_index \"$file_pattern_to_index\""
fi
if [[ "$file_list" != false ]]; then
    python_cmd+="   --files_to_index $file_list"
fi
if [[ "$filter_by_keywords" == true ]]; then
    python_cmd+="  --filter_by_keywords"
fi

# Execute the command
eval $python_cmd
#echo $python_cmd

