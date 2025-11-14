import os
import logging
import torch
import pandas as pd
from tqdm.auto import tqdm
import glob
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
here = os.path.dirname(os.path.abspath(__file__))


keywords = [
    'council',
    'policy',
    'town board',
    'votes',
    'member',
    'supervisor',
    'local',
    'proposal',
    'ordinance',
    'resolution',
    'meeting',
    'agenda',
    'minutes',
    'public hearing',
    'public comment',
    'task force',
    'committee',
    'board',
    'commission',
    'zoning',
    'planning',
]

def create_index_transcript(transcript_df, transcript_file_path):
    fp, f = os.path.dirname(transcript_file_path), os.path.basename(transcript_file_path)
    f_id = f.replace('.transcribed.json', '')
    schedule_f_dir = fp.replace('_transcribed_files', '_videos')
    schedule_f_name = f'{f_id}.schedule.csv'
    schedule_f_path = os.path.join(schedule_f_dir, schedule_f_name)
    if os.path.exists(schedule_f_path):
        schedule_df = pd.read_csv(schedule_f_path)
        if 'time' not in schedule_df.columns.tolist():
            schedule_df = pd.concat([
                pd.DataFrame({'time': [0], 'title': 'Start'}),
                pd.DataFrame({'time': [transcript_df['end'].max()], 'title': 'End'}),
            ])
        if not (0  in schedule_df['time']):
            schedule_df = pd.concat([
                pd.DataFrame({'time': [0], 'title': 'Start'}),
                schedule_df,
            ])
        if not (transcript_df['end'].max() in schedule_df['time']):
            schedule_df = pd.concat([
                schedule_df,
                pd.DataFrame({'time': [transcript_df['end'].max()], 'title': 'End'}),
            ])
        schedule_df = schedule_df.reset_index(drop=True).drop_duplicates('time')


        bins = schedule_df['time'].sort_values().tolist()
        transcript_df['meeting_segment'] = transcript_df['start'].pipe(lambda s: pd.cut(s, bins))
        transcript_df['meeting_segment'] = (
            transcript_df['meeting_segment']
                .apply(lambda x: x.left)
                .map(schedule_df.set_index('time')['title'].fillna('null').to_dict())
        )
    else:
        transcript_df['meeting_segment'] = 'None'

    return (
        f_id +
             '___' + 
             transcript_df['meeting_segment'].astype(str) + 
             '___' + 
             transcript_df.index.astype(str)
        )


def read_file(file_path, id_col):
    is_transcript = False
    if 'transcribed.json' in file_path:
        is_transcript = True

    if (file_path.endswith('.json') or file_path.endswith('.jsonl')):
        if is_transcript:
            f = json.load(open(file_path))
            df = pd.DataFrame(f['segments'])
        else:
            df = pd.read_json(file_path, lines=True)

    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if len(df) == 0:
        logging.warning(f"No data found in {file_path}")
        return None

    df['filename'] = file_path
    if id_col not in df.columns:
        if is_transcript:
            df[id_col] = create_index_transcript(df, file_path)
        else:
            df[id_col] = df['filename'] + df.index.astype(str)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name",
        type=str,
        help="Name of the index to create",
        default="new-index",
    )
    parser.add_argument(
        "--file_to_index",
        type=str,
        help="Path to the file(s) containing the documents to index",
        default=None,
    )
    parser.add_argument(
        "--file_pattern_to_index",
        type=str,
        help="Pattern of the filepath containing the documents to index",
        default=None,
    )
    parser.add_argument(
        "--files_to_index",
        type=str,
        nargs='+',
        help="Path to the file(s) containing the documents to index",
        default=None,
    )
    parser.add_argument(
        "--file_for_inference",
        type=str,
        help="Path to the file containing the queries to search",
        default=None,
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default="nvidia/NV-Embed-v1",  # "sentence-transformers/all-MiniLM-L6-v2", #
        help="The model to use for generating embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for inference"
    )
    parser.add_argument(
        "--id_col",
        type=str,
        help="Name of the column containing the unique ID for each document to index",
        default="article_url",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        help="Name of the column containing the text",
        default="article_text",
    )

    # defaults and configs
    parser.add_argument(
        "--retriv_cache_dir",
        type=str,
        default=here,
        help="Path to the directory containing indices"
    )
    parser.add_argument(
        "--huggingface_cache_dir",
        type=str,
        default='/project/jonmay_231/spangher/huggingface_cache',
        help="Path to the directory containing HuggingFace cache"
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=None,  # 4096
        help="The dimension of the embeddings"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,  # 32768,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--batch_size_to_index",
        type=int,
        help="Batch size for indexing",
        default=1,
    )
    parser.add_argument(
        "--start_index",
        type=int,
        help="Index to start from",
        default=0,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        help="Index to end at",
        default=-1,
    )
    parser.add_argument(
        '--date_col',
        type=str,
        help="Name of the column containing the date",
        default='article_publish_date',
    )
    parser.add_argument(
        '--filter_by_keywords',
        action='store_true',
        help="Filter the articles by keywords"
    )
    parser.add_argument(
        '--filter_with_lr',
        action='store_true',
        help="Use a trained logistic regression model to filter"
    )

    args = parser.parse_args()

    # pretty print args
    args_str = '\n'.join([f"{k}:\t\t{v}" for k, v in vars(args).items()])
    logging.info(f"""Arguments: \n{args_str}""")
    hf_cache_dir = args.huggingface_cache_dir
    logging.info(f"Setting environment variables: HF_HOME={hf_cache_dir}, HF_TOKEN_PATH={hf_cache_dir}/token")

    os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['HF_TOKEN_PATH'] = f'{hf_cache_dir}/token'
    os.environ['HF_TOKEN'] = open(f'{hf_cache_dir}/token').read().strip()

    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    from retriv import DenseRetriever, SparseRetriever

    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    files_to_index = []
    if args.file_pattern_to_index is not None:
        logging.info(f"Indexing files matching pattern: {args.file_pattern_to_index}")
        files_to_index += glob.glob(args.file_pattern_to_index)
    if args.files_to_index is not None:
        files_to_index += args.files_to_index
    if args.file_to_index is not None:
        files_to_index.append(args.file_to_index)

    if len(files_to_index) > 0:
        # read files to index
        df_to_index = []
        for f in tqdm(files_to_index, desc="Reading files to index"):
            df = read_file(f, args.id_col)
            if df is None:
                continue
            df = (
                df.drop_duplicates(args.id_col)
                    .loc[lambda df: df[args.text_col].str.strip().notnull()]
            )
            df_to_index.append(df)
        df_to_index = pd.concat(df_to_index)
        if args.filter_by_keywords:
            df_to_index = df_to_index.loc[lambda df: df[args.text_col].str.contains('|'.join(keywords), case=False)]
            args.index_name = f"{args.index_name}__keyword-filtered"

        if args.filter_with_lr:
            import pickle
            lr_model = pickle.load(open('lr-model.pkl', 'rb'))
            y_preds = lr_model.predict_proba(df_to_index[args.text_col].fillna(''))[:, 1]
            y_true = (y_preds > .65)
            df_to_index = df_to_index[y_true]
            args.index_name = f"{args.index_name}__lr-filtered"

        if args.end_index > 0:
            df_to_index = df_to_index.sort_values(args.id_col)
            df_to_index = df_to_index.iloc[args.start_index:args.end_index]
            min_date = df_to_index[args.date_col].min()
            max_date = df_to_index[args.date_col].max()
            args.index_name = f"{args.index_name}__{min_date}_{max_date}"

        data_to_index = (
            df_to_index
            [[args.id_col, args.text_col]]
            .rename(columns={args.id_col: 'id', args.text_col: 'text'})
            .to_dict(orient='records')
        )

        # set up index
        if (args.embedding_model == 'bm25') or (args.embedding_model == 'tf-idf'):
            retriever = SparseRetriever(
                index_name=args.index_name + '__sparse',
                model=args.embedding_model,
            )
            retriever = retriever.index(
                collection=data_to_index,
                show_progress=True,  # Default value
            )
        else:
            retriever = DenseRetriever(
                index_name=args.index_name,
                model=args.embedding_model,
                normalize=True,
                max_length=args.max_seq_length,
                embedding_dim=args.embedding_dim,
                device=args.device,
                use_ann=True,
                transformers_cache_dir=hf_cache_dir
            )

            retriever = retriever.index(
                collection=data_to_index,  # File kind is automatically inferred
                batch_size=args.batch_size_to_index,  # Default value
                show_progress=True,  # Default value
            )


"""
python retriv_index_files.py \
    --file_pattern_to_index "../data/seattle_videos/*.schedule.csv" \
    --text_col title \
    --id_col id \
    --index_name seattle_agenda \
    --embedding_model Salesforce/SFR-Embedding-2_R
"""