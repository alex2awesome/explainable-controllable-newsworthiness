import os
import logging
import torch
import pandas as pd
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

    args = parser.parse_args()
    hf_cache_dir = args.huggingface_cache_dir
    logging.info(f"Setting environment variables: HF_HOME={hf_cache_dir}, HF_TOKEN_PATH={hf_cache_dir}/token")

    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['HF_TOKEN_PATH'] = f'{hf_cache_dir}/token'
    os.environ['HF_TOKEN'] = open(f'{hf_cache_dir}/token').read().strip()

    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    from dense_retriever import MyDenseRetriever

    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    if args.file_to_index is not None:
        # read files to index
        df_to_index = []
        for f in args.file_to_index:
            df = pd.read_json(f, lines=True)
            df = (
                df.drop_duplicates(args.id_col)
                    .loc[lambda df: df[args.text_col].str.strip().notnull()]
            )
            df_to_index.append(df)
        df_to_index = pd.concat(df_to_index)
        if args.filter_by_keywords:
            df_to_index = df_to_index.loc[lambda df: df[args.text_col].str.contains('|'.join(keywords), case=False)]
            args.index_name = f"{args.index_name}__keyword-filtered"

        if args.end_index > 0:
            df_to_index = df_to_index.sort_values(args.id_col)
            df_to_index = df_to_index.iloc[args.start_index:args.end_index]
            min_date = df_to_index[args.date_col].min()
            max_date = df_to_index[args.date_col].max()
            args.index_name = f"{args.index_name}__{min_date}_{max_date}"

        # set up index
        dr = MyDenseRetriever(
            index_name=args.index_name,
            model=args.embedding_model,
            normalize=True,
            max_length=args.max_seq_length,
            embedding_dim=args.embedding_dim,
            device=args.device,
            use_ann=True,
        )

        dr = dr.index(
            collection=(
                df_to_index
                    [[args.id_col, args.text_col]]
                    .rename(columns={args.id_col: 'id', args.text_col: 'text'})
                    .to_dict(orient='records')
            ),  # File kind is automatically inferred
            batch_size=args.batch_size_to_index,  # Default value
            show_progress=True,  # Default value
        )
