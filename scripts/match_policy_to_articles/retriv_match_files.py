import functools
import re
import pandas as pd
import logging
import torch
import os
import glob
from urllib.parse import quote, unquote
from retriv import DenseRetriever, Encoder, SparseRetriever
import datetime
import json
from collections import defaultdict
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
here = os.path.dirname(os.path.abspath(__file__))
DR_THRESHOLD_POLICIES = 0.6
SR_THRESHOLD_POLICIES = 20
DR_THRESHOLD_NEWS = 0.4
SR_THRESHOLD_NEWS = 15
NUM_WORDS_PER_TRANSCRIPT = 50
USE_AGENDA_FOR_QUERY = False

AGENDA_MATCHING_INSTRUCTION = "Instruct: Match the vague policy description to discussion of the policy in this meeting. Query: {policy_title}"

city_name_to_newspaper_mapper = {
    # '': 'alexandria-times-articles-sans-html.jsonl',
    'jacksonville': ['florida-times-articles-sans-html.jsonl', 'jax-daily-record-articles-sans-html.jsonl'],
    'seattle': ['seattle-times-articles-sans-html.jsonl'],
    'durham': [
        'raleigh-news-and-observer-articles-sans-html.jsonl', 'durham-herald-articles-sans-html.jsonl'
    ],
    'denver': ['denver-post-articles-sans-html.jsonl'],
    'fortworth': ['fort-worth-star-telegram-articles-sans-html.jsonl'],
    'newark': ['star-ledger-articles-sans-html.jsonl'],
}

def get_news_index_with_date(city_name):
    news_files = city_name_to_newspaper_mapper[city_name]
    news_indices = []
    for news_file in news_files:
        news_df = pd.read_json(f'../data/news_articles/{news_file}', lines=True)
        news_index = news_df[['article_url', 'article_publish_date']]
        # get article publish date
        date_from_url_1 = (
            news_df['article_url']
                .fillna('')
                .apply(lambda x: re.search(r'\d{4}/\d{2}/\d{2}', x))
                .apply(lambda x: x[0] if x is not None else None).to_frame('article_url_date')
        )
        date_from_url_2 = (
            news_df['article_url']
                .fillna('')
                .apply(lambda x: re.search(r'\d{4}/\d{4}', x))
                .apply(lambda x: x[0] if x is not None else None).to_frame('article_url_date')
        )
        date_from_image_1 = (
            news_df['article_top_image']
                .fillna('')
                .apply(lambda x: re.search(r'20\d{6}', x)).apply(lambda x: x[0] if x is not None else None)
                .to_frame('article_image_date_1')
        )
        date_from_image_2 = (
            news_df['article_top_image']
                .fillna('')
                .apply(lambda x: re.search(r'20\d{2}/\d{2}', x))
                .apply(lambda x: x[0] if x is not None else None).to_frame('article_image_date_2')
        )
        url_dates = pd.concat([date_from_url_1, date_from_url_2, date_from_image_1, date_from_image_2], axis=1)
        url_dates['article_image_date_1'] = pd.to_datetime(url_dates['article_image_date_1'], format='%Y%m%d', errors='coerce')
        url_dates = url_dates.stack().groupby(level=0).first().reindex(url_dates.index)
        url_dates = pd.to_datetime(url_dates, errors='coerce').dt.date
        news_index['article_publish_date'] = pd.to_datetime(
            news_index['article_publish_date'],
            errors='coerce',
            utc=True,
            format='mixed'
        ).dt.date
        news_index['article_publish_date'] = pd.to_datetime(
            news_index['article_publish_date'].combine_first(url_dates), format='mixed'
        )
        news_indices.append(news_index)
    news_index_df = pd.concat(news_indices)
    logging.info(f"Loaded {news_index_df.shape[0]} news articles for: {city_name}")
    logging.info(f"# of articles with null DT: {news_index_df['article_publish_date'].isnull().value_counts().to_dict()}")
    logging.info(
        f"% articles with null DT: {news_index_df['article_publish_date'].isnull().value_counts().pipe(lambda s: s/s.sum()).to_dict()}")
    news_index_df = news_index_df.loc[lambda df: df['article_publish_date'].notnull()].set_index('article_publish_date').sort_index()
    return news_index_df


def get_all_indices(args):
    if args.agenda_index is None:
        args.agenda_index = f"{args.city_name}_agenda"
    if args.public_comment_index is None:
        args.public_comment_index = f"{args.city_name}_transcript"
    if args.news_article_index_pattern is None:
        args.news_article_index_pattern = f"{args.city_name}_index__keyword-filtered*"
        args.news_article_index_pattern_sparse = f"{args.city_name}_index__keyword-filtered_*__sparse"

    logging.info(f"Loading encoder for: {args.city_name}")
    encoder = Encoder(index_name=None, model=args.encoder_model, device=args.device, transformers_cache_dir=args.huggingface_cache_dir)

    # load agenda index
    logging.info(f"Loading agenda index for: {args.city_name}")
    sr_agenda = SparseRetriever.load(args.agenda_index + '__sparse')
    dr_agenda = DenseRetriever.load(args.agenda_index, skip_encoder_loading=True, transformers_cache_dir=args.huggingface_cache_dir)
    dr_agenda.encoder = encoder

    # load public comment index
    logging.info(f"Loading speaker index for: {args.city_name}")
    sr_public_comment = SparseRetriever.load(args.public_comment_index + '__sparse')
    dr_public_comment = DenseRetriever.load(args.public_comment_index, skip_encoder_loading=True, transformers_cache_dir=args.huggingface_cache_dir)
    dr_public_comment.encoder = encoder

    # load news index
    dense_news_indices = glob.glob(os.path.join(args.retriv_cache_dir, 'collections', args.news_article_index_pattern))
    dense_news_indices = list(filter(lambda x: 'sparse' not in x, dense_news_indices))
    sparse_news_indices = glob.glob(os.path.join(args.retriv_cache_dir, 'collections', args.news_article_index_pattern_sparse))
    dr_news_articles, sr_news_articles = [], []
    for index_name in dense_news_indices:
        index_name = os.path.basename(index_name)
        logging.info(f"Loading dense news index: {index_name}")
        dr = DenseRetriever.load(index_name, skip_encoder_loading=True, transformers_cache_dir=args.huggingface_cache_dir)
        dr.encoder = encoder
        dr_news_articles.append(dr)
    for index_name in sparse_news_indices:
        index_name = os.path.basename(index_name)
        logging.info(f"Loading sparse news index: {index_name}")
        sr = SparseRetriever.load(index_name)
        sr_news_articles.append(sr)

    return sr_agenda, dr_agenda, sr_public_comment, dr_public_comment, dr_news_articles, sr_news_articles, encoder


def get_meeting_df(city_name):
    # if city_name == 'seattle':
    #     policy_df = (pd.read_csv('../data/seattle-meeting-info.csv')
    #                  .rename(columns={
    #                             'Record No': 'File #',
    #                             'Record No_href': 'File #_href'
    #                  }))
    #     meeting_df = (pd.read_csv('../data/seattle-meeting-dates.csv')
    #                   .rename(columns={
    #                         'Seattle Channel': 'Video',
    #                         'Seattle Channel_href': 'video_url'
    #                   }))
    #     policy_df = policy_df.merge(
    #         meeting_df[['Meeting Details_href', 'video_url']], left_on='key', right_on='Meeting Details_href'
    #     )
    #     policy_df
    # else:
    policy_df = pd.read_csv(f'../data/{city_name}__policy_info.csv')
    meeting_df = pd.read_csv(f'../data/{city_name}__meeting_info.csv')
    policy_df = policy_df.merge(
        meeting_df[['Meeting Details_href', 'Meeting Date']], left_on='key', right_on='Meeting Details_href'
    )

    policy_df['Meeting Date'] = pd.to_datetime(policy_df['Meeting Date'])
    policy_df = policy_df.set_index(['Meeting Date']).sort_index()
    if 'Name' not in policy_df:
        policy_df['Name'] = ''
    policy_df['Name'] = policy_df['Name'].fillna('')
    policy_df['Title'] = policy_df['Title'].fillna('')
    policy_df = policy_df.rename(columns={'Record No_href': 'File #'})
    return policy_df


def get_relevant_indices(meeting_filters, se_index_df):
    meeting_filter_df = meeting_filters.to_frame('quoted_url')
    desired_meetings = se_index_df.merge(meeting_filter_df, on='quoted_url')
    return desired_meetings['index_key'].tolist()


def get_relevant_meeting_indices_for_window(row, policy_df, full_se_index, time_window, city_name):
    meeting_date = row.name #['Meeting Date']
    start_time, end_time = (meeting_date - datetime.timedelta(days=time_window)), (meeting_date + datetime.timedelta(days=time_window))
    # acceptable_meetings = policy_df.loc[lambda df: df['Meeting Date'].between(start_time, end_time)]
    acceptable_meetings = policy_df.loc[start_time: end_time]
    video_urls = acceptable_meetings['video_url'].dropna().drop_duplicates()

    quoted_urls = list(map(lambda x: quote(x, safe=''), video_urls))
    relevant_agenda_indices = []
    transcribed_dfs = []
    for quoted_url in quoted_urls:
        transcribed_video_csv_path = f'../data/{city_name}_transcribed_files/{quoted_url}.transcribed.json'
        transcribed_df = pd.DataFrame(json.load(open(transcribed_video_csv_path))['segments'])
        transcribed_dfs.append(transcribed_df)
        # schedule_csv_path = f'../data/{city_name}_videos/{quoted_url}.schedule.csv'
        relevant_agenda_indices += list(filter(lambda x: quoted_url in x, full_se_index))
    return relevant_agenda_indices


def get_transcribed_meeting_text_by_timestamp(meeting_item, policy_df):
    # '../data/denver_videos/https%3A%2F%2Fdenver.granicus.com%2Fplayer%2Fclip%2F15993%3Fview_id%3D180%26redirect%3Dtrue.schedule.csv0'
    # parse the video url to get the numbers at the end of the string:
    meeting_file, index_name = meeting_item['id'].split('.schedule.csv')
    orig_video_url = unquote(meeting_file.split('/')[-1])
    transcript_file = f"{meeting_file.replace('_videos', '_transcribed_files')}.transcribed.json"
    meeting_file = f'{meeting_file}.schedule.csv'
    index_name = int(index_name)
    schedule_df = pd.read_csv(meeting_file, index_col=0)
    if not os.path.exists(transcript_file):
        return

    transcribed_df = pd.DataFrame(json.load(open(transcript_file))['segments'])
    if ('start' not in transcribed_df.columns) or ('end' not in transcribed_df.columns):
        return

    # print('----------------------------')
    # print(f"Meeting file: {meeting_file}, index_name: {index_name}")
    # print(transcribed_df)
    # print('----------------------------')
    index_name = int(index_name)
    start_time = schedule_df.loc[index_name]['time']
    if index_name == schedule_df.index.max():
        end_time = transcribed_df.iloc[-1]['end']
    else:
        end_time = schedule_df.loc[index_name + 1]['time']
    #
    meeting_segment = transcribed_df.loc[lambda df: (df['start'] > start_time) & (df['start'] < end_time)]
    meeting_segment_text = ' '.join(meeting_segment['text'].str.strip())
    if 'speaker' in meeting_segment.columns:
        meeting_segment_num_speakers = meeting_segment['speaker'].drop_duplicates().shape[0]
    else:
        meeting_segment_num_speakers = None

    # meeting_date = policy_df.loc[lambda df: df['video_url'] == orig_video_url]['Meeting Date'].iloc[0]
    meeting_date = policy_df.loc[lambda df: df['video_url'] == orig_video_url].index[0]
    return {
        'transcribed_text': meeting_segment_text,
        'number_of_speakers': meeting_segment_num_speakers,
        'meeting_date': meeting_date
    }


def get_all_ids_by_start_and_end_dates(policy_df, agenda_index_df, transcript_index_df, end_date, start_date=None):
    if start_date is None:
        # start_date = policy_df['Meeting Date'].min()
        start_date = policy_df.index.min()
    # recent_meetings = policy_df.loc[lambda df: df['Meeting Date'].between(start_date, end_date)]
    recent_meetings = policy_df.loc[start_date: end_date]
    meeting_filter_strings = (
        recent_meetings['video_url']
        .dropna().drop_duplicates()
        .apply(lambda x: quote(x, safe=''))
    )
    recent_schedule_items = get_relevant_indices(meeting_filter_strings, agenda_index_df)
    recent_transcript_items = get_relevant_indices(meeting_filter_strings, transcript_index_df)
    return recent_schedule_items, recent_transcript_items


def get_transcribed_text_package(transcribed_item, policy_df):
    # 'https%3A%2F%2Fdenver.granicus.com%2Fplayer%2Fclip%2F14461%3Fview_id%3D180%26redirect%3Dtrue___Start___0'
    id = transcribed_item['id']
    video_url, meeting_segment, _ = id.split('___')
    video_url = unquote(video_url)
    # meeting_date = policy_df.loc[lambda df: df['video_url'] == video_url]['Meeting Date'].iloc[0]
    meeting_date = policy_df.loc[lambda df: df['video_url'] == video_url].index[0]
    return {
        'transcribed_text': transcribed_item['text'],
        'meeting_date': meeting_date,
        'meeting_segment': meeting_segment
    }


def search_and_package(sr, dr, policy_to_search, target_items, policy_df, trial_name=None, key=None):
    if len(target_items) == 0:
        print(f"No aligned search results for: {key}, trial: {trial_name}")
        return []

    if (len(policy_to_search) > 50) and not (dr.index_name == 'seattle_transcript'):
        # if isinstance(se, DenseRetriever):
        # policy_embedding = encoder.encode(policy_to_search)
        # results = se.search(encoded_query=policy_embedding, include_id_list=target_items)
        # print(f"Searching for: {policy_to_search}")
        # print(f"filtering on: {target_items}")
        try:
            results = dr.search(query=policy_to_search, include_id_list=target_items)
        except Exception as e:
            print(f"Error searching for: {policy_to_search}, error: {str(e)}")
            return []
        threshold = DR_THRESHOLD_POLICIES
    else:
        results = sr.search(policy_to_search, include_id_list=target_items)
        threshold = SR_THRESHOLD_POLICIES
    if 'transcript' in trial_name:
        package_func = get_transcribed_text_package
    else:
        package_func = get_transcribed_meeting_text_by_timestamp

    ##
    results = list(filter(lambda x: x['score'] > threshold, results))
    results = list(map(lambda x: package_func(x, policy_df), results))
    return list(filter(lambda x: x is not None, results))


def make_news_article_query(policy_row):
    def take_first_n_words_of_transcript(d, n=NUM_WORDS_PER_TRANSCRIPT):
        return ' '.join(d['transcribed_text'].split()[:n])

    def combine_all_transcripts(r):
        segments = list(map(take_first_n_words_of_transcript, r))
        return '... '.join(segments)

    output_text = f"Policy item: ```{policy_row['policy_title']}```"
    exact_agenda_results = policy_row.get('exact_agenda_results') or []
    if (len(exact_agenda_results) > 0) and (USE_AGENDA_FOR_QUERY):
        exact_agenda_text = f"Meeting Discussion: ```{combine_all_transcripts(exact_agenda_results)}```"
        output_text += f"\n{exact_agenda_text}"

    exact_transcript_results = policy_row.get('exact_transcript_results') or []
    if len(exact_transcript_results) > 0:
        exact_transcript_text = f"Similar topics discussed in meeting: ```{combine_all_transcripts(exact_transcript_results)}```"
        output_text += f"\n{exact_transcript_text}"

    if (len(exact_agenda_results) > 0) or (len(exact_transcript_results) > 0):
        return output_text

    time_window_agenda_results = policy_row.get('time_window_agenda_results') or []
    if (len(time_window_agenda_results) > 0) and (USE_AGENDA_FOR_QUERY):
        time_window_agenda_text = f"Similar Meeting Discussions: ```{combine_all_transcripts(time_window_agenda_results)}```"
        output_text += f"\n{time_window_agenda_text}"

    time_window_transcript_results = policy_row.get('time_window_transcript_results') or []
    if len(time_window_transcript_results) > 0:
        time_window_transcript_text = f"Similar topics in other meetings: ```{combine_all_transcripts(time_window_transcript_results)}```"
        output_text += f"\n{time_window_transcript_text}"

    if (len(time_window_agenda_results) > 0) or (len(time_window_transcript_results) > 0):
        return output_text

    all_agenda_results = policy_row.get('all_agenda_results') or []
    if (len(all_agenda_results) > 0) and (USE_AGENDA_FOR_QUERY):
        all_agenda_text = f"Similar Meeting Discussions: ```{combine_all_transcripts(all_agenda_results)}```"
        output_text += f"\n{all_agenda_text}"

    all_transcript_results = policy_row.get('all_transcript_results') or []
    if len(all_transcript_results) > 0:
        all_transcript_text = f"Similar topics in other meetings: ```{combine_all_transcripts(all_transcript_results)}```"
        output_text += f"\n{all_transcript_text}"

    return output_text


def json_serialize_output(transcript_items):
    output = []
    for t in transcript_items:
        if 'meeting_date' in t:
            t['meeting_date'] = str(t['meeting_date'])
        output.append(t)
    return output


# pd.read_csv('../data/denver__policy_info.csv')
def add_args(parser):
    parser.add_argument(
        "--meeting_policy_input_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--policy_text_output_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--policy_and_news_article_output_file",
        type=str,
        default=None,
    )

    # index names
    parser.add_argument(
        '--city_name',
        type=str,
        default=None
    )
    parser.add_argument(
        "--agenda_index",
        type=str,
        default=None
    )
    parser.add_argument(
        "--public_comment_index",
        type=str,
        default=None
    )
    parser.add_argument(
        "--news_article_index_pattern",
        type=str,
        default=None
    )
    parser.add_argument(
        "--encoder_model",
        type=str,
        default='Salesforce/SFR-Embedding-2_R',
    )

    # model cache directories
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
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--match_policies_to_transcripts",
        action='store_true',
        help="Match policies to transcripts"
    )
    parser.add_argument(
        "--match_policies_to_news_articles",
        action='store_true',
        help="Match policies to news articles"
    )

    return parser


# python retriv_match_files.py \
#   --agenda_index seattle_agenda \
#   --public_comment_index seattle_public_comment \
#   --news_article_index_pattern denver_index__keyword-filtered_* \
#   --encoder_model 'Salesforce/SFR-Embedding-2_R'
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # input files
    parser = add_args(parser)

    args = parser.parse_args()
    logging.info(f"Running with arguments: {args}")

    hf_cache_dir = args.huggingface_cache_dir
    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
    os.environ['HF_TOKEN_PATH'] = f'{hf_cache_dir}/token'
    os.environ['HF_TOKEN'] = open(f'{hf_cache_dir}/token').read().strip()
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir

    # load the indices
    sr_agenda, dr_agenda, sr_transcript, dr_transcript, dr_news_articles, sr_news_articles, encoder = get_all_indices(args)
    agenda_index_df = pd.Series(sr_agenda.doc_index.index_keys).to_frame('index_key')
    agenda_index_df['quoted_url'] = agenda_index_df['index_key'].str.split('/').str.get(-1).str.split('.schedule').str.get(0)
    transcript_index_df = pd.Series(sr_transcript.doc_index.index_keys).to_frame('index_key')
    transcript_index_df['quoted_url'] = transcript_index_df['index_key'].str.split('___').str.get(0)

    # dr.doc_index.index
    policy_df = get_meeting_df(args.city_name)
    # for testing/debugging
    policy_df_to_iterate = policy_df.copy()
    if args.start_idx is not None:
        policy_df_to_iterate = policy_df_to_iterate.iloc[args.start_idx:]
    if args.num_rows is not None:
        policy_df_to_iterate = policy_df_to_iterate.iloc[:args.num_rows]

    if args.policy_text_output_file is None:
        args.policy_text_output_file = f'{args.city_name}-policies-matched-with-speakers-test.jsonl',
    if args.policy_and_news_article_output_file is None:
        args.policy_and_news_article_output_file = f'{args.city_name}-policies-matched-with-news-articles.jsonl',

    full_policy_text_output = []
    search_and_package = functools.partial(search_and_package, policy_df=policy_df)
    logging.info(f"Processing policy meetings for: {args.city_name}")

    ######################################################
    #
    # Part 1: Matching policies to transcript segments
    #
    ######################################################

    if args.match_policies_to_transcripts:
        output_file_handle = open(args.policy_text_output_file, 'w')
        for meeting_date, target_meeting in tqdm(policy_df_to_iterate.iterrows(), total=len(policy_df_to_iterate), desc='Matching policy meetings'):
            policy_to_search = target_meeting['Title']
            name = target_meeting['Name']
            if pd.isnull(name):
                name = ''
            policy_to_search = (name + ' ' + policy_to_search).strip()

            policy_with_instruction = AGENDA_MATCHING_INSTRUCTION.format(policy_title=policy_to_search)
            output_packet = {
                'key': target_meeting['key'],
                'file #': target_meeting['File #'],
                # 'Meeting Date': str(target_meeting['Meeting Date']),
                'Meeting Date': str(meeting_date),
                'policy_title': policy_to_search,
            }

            # for the exact meeting, for recent meetings, and for meetings of all time:
            # --------------------------------------------------------------------------------------------
            # 1. Do a first pass where you match the policy title to the row in the agenda index.
            #    Then, use the timestamps in the index to segment the video transcription.
            #    This is so we have speech corresponding to each piece of text.
            # --------------------------------------------------------------------------------------------
            # 2. Match each speaker's dialogue in the transcribed video file to the policy title.
            # --------------------------------------------------------------------------------------------
            if pd.notnull(target_meeting['video_url']):
                target_key_str = quote(target_meeting['video_url'], safe='')
                # print(f"Processing: {target_key_str}")
                target_schedule_items = agenda_index_df.loc[lambda df: df['quoted_url'] == target_key_str]['index_key'].tolist()
                target_transcript_items = transcript_index_df.loc[lambda df: df['quoted_url'] == target_key_str]['index_key'].tolist()

                # get policy discussion by meeting
                exact_agenda_results = search_and_package(
                    sr_agenda, dr_agenda, policy_to_search, target_schedule_items, key=target_key_str,
                    trial_name='exact_meeting_agenda'
                )
                exact_transcript_results = search_and_package(
                    sr_transcript, dr_transcript, policy_with_instruction, target_transcript_items, key=target_key_str,
                    trial_name='exact_meeting_transcript'
                )

                # get related policy discussion get recent meetings
                # start_date = (target_meeting['Meeting Date'] - datetime.timedelta(days=7))
                start_date = (meeting_date - datetime.timedelta(days=7))
                recent_schedule_items, recent_transcript_items = get_all_ids_by_start_and_end_dates(
                    policy_df, end_date=meeting_date, start_date=start_date,
                    agenda_index_df=agenda_index_df, transcript_index_df=transcript_index_df
                )

                time_window_agenda_results = search_and_package(
                    sr_agenda, dr_agenda, policy_to_search, recent_schedule_items, key=target_key_str,
                    trial_name='recent_meeting_agenda'
                )
                time_window_transcript_results = search_and_package(
                    sr_transcript, dr_transcript, policy_with_instruction, recent_transcript_items,
                    key=target_key_str, trial_name='recent_meeting_transcript'
                )

                # get related policy discussion in all prior meetings
                all_prior_schedule_items, all_prior_transcript_items = get_all_ids_by_start_and_end_dates(
                    policy_df, end_date=meeting_date, agenda_index_df=agenda_index_df,
                    transcript_index_df=transcript_index_df
                )
                all_agenda_results = search_and_package(
                    sr_agenda, dr_agenda, policy_to_search, all_prior_schedule_items, key=target_key_str,
                    trial_name='recent_meeting_agenda'
                )
                all_transcript_results = search_and_package(
                    sr_transcript, dr_transcript, policy_with_instruction, all_prior_transcript_items,
                    key=target_key_str, trial_name='recent_meeting_transcript'
                )

                output_packet['exact_agenda_results'] = json_serialize_output(exact_agenda_results)
                output_packet['exact_transcript_results'] = json_serialize_output(exact_transcript_results)
                output_packet['time_window_agenda_results'] = json_serialize_output(time_window_agenda_results)
                output_packet['time_window_transcript_results'] = json_serialize_output(time_window_transcript_results)
                output_packet['all_agenda_results'] = json_serialize_output(all_agenda_results)
                output_packet['all_transcript_results'] = json_serialize_output(all_transcript_results)

            full_policy_text_output.append(output_packet)
            output_file_handle.write(json.dumps(output_packet) + '\n')

        full_policy_text_df = pd.DataFrame(full_policy_text_output)


    ######################################################
    #
    # Part 2: Matching policies to news articles
    #
    ######################################################

    if args.match_policies_to_news_articles:
        output_file_handle = open(args.policy_and_news_article_output_file, 'w')
        full_policy_text_df = pd.read_json(args.policy_text_output_file, lines=True)
        full_policy_text_df['Meeting Date'] = pd.to_datetime(full_policy_text_df['Meeting Date'])
        news_date_index = get_news_index_with_date(args.city_name).drop_duplicates()
        news_index_to_date = news_date_index.reset_index().set_index('article_url')
        logging.info(f"Processing news files for: {args.city_name}")
        for _, row in tqdm(full_policy_text_df.iterrows(), total=len(full_policy_text_df), desc='Matching policy meetings to news articles'):
            # match the policy/meeting text to the news article
            query = make_news_article_query(policy_row=row)
            # print(query)
            task_description = "Was this policy written about in this news article?"
            query_to_embed = f'Instruct: {task_description}\nQuery: {query}'
            encoded_query = encoder.encode(query_to_embed)
            start_date = row['Meeting Date'] - datetime.timedelta(days=30)
            end_date = row['Meeting Date']
            future_date = row['Meeting Date'] + datetime.timedelta(days=30)
            recent_past_news_articles = (
                news_date_index
                    # .loc[lambda df: df['article_publish_date'].between(start_date, end_date)]
                    .loc[start_date: end_date]
                    ['article_url'].tolist()
            )
            # all_past_news_articles = news_date_index.loc[lambda df: df['article_publish_date'] < end_date]['article_url'].tolist()
            # future_news_articles = news_date_index.loc[lambda df: df['article_publish_date'] > future_date]['article_url'].tolist()
            all_past_news_articles = news_date_index.loc[:end_date]['article_url'].tolist()
            future_news_articles = news_date_index.loc[future_date:]['article_url'].tolist()

            ##
            news_articles = defaultdict(list)
            for (name, url_filter_list) in [
                ('dense_recent_past', recent_past_news_articles),
                ('dense_all_past', all_past_news_articles),
                ('dense_future', future_news_articles)
            ]:
                for dr_news_article in dr_news_articles:
                    candidate_articles = dr_news_article.search(
                        encoded_query=encoded_query,
                        include_id_list=url_filter_list
                    )
                    output_candidate_articles = []
                    for candidate_article in candidate_articles:
                        if candidate_article['score'] > DR_THRESHOLD_NEWS:
                            if candidate_article['id'] in news_index_to_date.index:
                                dt = news_index_to_date.loc[candidate_article['id']]['article_publish_date']
                                candidate_article['article_publish_date'] = str(dt)
                            candidate_article['score'] = float(candidate_article['score'])
                            output_candidate_articles.append(candidate_article)
                    news_articles[name].extend(output_candidate_articles)

            for (name, url_filter_list) in [
                ('sparse_recent_past', recent_past_news_articles),
                ('sparse_all_past', all_past_news_articles),
                ('sparse_future', future_news_articles)
            ]:
                for sr_news_article in sr_news_articles:
                    candidate_articles = sr_news_article.search(
                        query=row['policy_title'],
                        include_id_list=url_filter_list
                    )
                    output_candidate_articles = []
                    for candidate_article in candidate_articles:
                        if candidate_article['score'] > SR_THRESHOLD_NEWS:
                            if candidate_article['id'] in news_index_to_date.index:
                                dt = news_index_to_date.loc[candidate_article['id']]['article_publish_date']
                                candidate_article['article_publish_date'] = str(dt)
                            candidate_article['score'] = float(candidate_article['score'])
                            output_candidate_articles.append(candidate_article)
                    news_articles[name].extend(output_candidate_articles)

            row['Meeting Date'] = str(row['Meeting Date'])
            output_dict = row.to_dict()
            output_dict['news_articles'] = news_articles
            output_file_handle.write(json.dumps(output_dict) + '\n')


"""
import os
from retriv import SparseRetriever, DenseRetriever
import pandas as pd 
from urllib.parse import quote
import datetime
import json
import attrdict
from tqdm.auto import tqdm

hf_cache_dir = '/project/jonmay_231/spangher/huggingface_cache'
here = os.getcwd()
os.environ['RETRIV_BASE_PATH'] = here
AGENDA_MATCHING_INSTRUCTION = "Instruct: Match the vague policy description to discussion of the policy in this meeting. Query: {policy_title}"
city_name = 'jacksonville'
args = attrdict.AttrDict(dict(
    meeting_policy_input_file=None, 
    policy_text_output_file='jacksonville-policies-matched-with-speakers-test.jsonl',
    policy_and_news_article_output_file='jacksonville-policies-matched-with-news-articles.jsonl',
    city_name='jacksonville', 
    encoder_model='Salesforce/SFR-Embedding-2_R', 
    retriv_cache_dir='/project/jonmay_231/spangher/Projects/explainable-controllable-newsworthiness/non_sf_policies/scripts', 
    huggingface_cache_dir='/project/jonmay_231/spangher/huggingface_cache', 
    device='cuda', 
    start_idx=1000,
    num_rows=20, 
    match_policies_to_transcripts=True, 
    match_policies_to_news_articles=True, agenda_index=None, public_comment_index=None, news_article_index_pattern=None,
))

sr_agenda, dr_agenda, sr_transcript, dr_transcript, dr_news_articles, sr_news_articles, encoder = (
    retriv_match_files.get_all_indices(args)
)
agenda_index_df = pd.Series(sr_agenda.doc_index.index_keys).to_frame('index_key')
agenda_index_df['quoted_url'] = agenda_index_df['index_key'].str.split('/').str.get(-1).str.split('.schedule').str.get(0)
transcript_index_df = pd.Series(sr_transcript.doc_index.index_keys).to_frame('index_key')
transcript_index_df['quoted_url'] = transcript_index_df['index_key'].str.split('___').str.get(0)

policy_df = retriv_match_files.get_meeting_df(args.city_name)
policy_df = policy_df.iloc[args.start_idx:]
policy_df = policy_df.iloc[:args.num_rows]
full_policy_text_output = []

# if args.match_policies_to_transcripts:
output_file_handle = open(args.policy_text_output_file, 'w')
for meeting_date, target_meeting in tqdm(policy_df.iterrows(), total=len(policy_df), desc='Matching policy meetings'):
    policy_to_search = target_meeting['Title']
    name = target_meeting['Name']
    if pd.isnull(name):
        name = ''
    policy_to_search = (name + ' ' + policy_to_search).strip()

    policy_with_instruction = AGENDA_MATCHING_INSTRUCTION.format(policy_title=policy_to_search)
    if pd.notnull(target_meeting['video_url']):
        target_key_str = quote(target_meeting['video_url'], safe='')
        target_schedule_items = agenda_index_df.loc[lambda df: df['quoted_url'] == target_key_str]['index_key'].tolist()
        target_transcript_items = transcript_index_df.loc[lambda df: df['quoted_url'] == target_key_str]['index_key'].tolist()

        # get policy discussion by meeting
        exact_agenda_results = retriv_match_files.search_and_package(
            sr_agenda, dr_agenda, policy_to_search, target_schedule_items, key=target_key_str,
            trial_name='exact_meeting_agenda', policy_df=policy_df
        )
        exact_transcript_results = retriv_match_files.search_and_package(
            sr_transcript, dr_transcript, policy_with_instruction, target_transcript_items, key=target_key_str,
            trial_name='exact_meeting_transcript', policy_df=policy_df
        )

"""


"""
server experimentation:
# 1. first goal is to load the sparse retriever and use it to connect policy titles to agenda-list items 
# 2. second goal is to use these list items to 
 
# pip install git+https://github.com/alex2awesome/retriv.git --force-reinstall --no-deps
conda activate retriv-py39
ipython

import os
from retriv import SparseRetriever, DenseRetriever
import pandas as pd 
from urllib.parse import quote
import datetime
import json

hf_cache_dir = '/project/jonmay_231/spangher/huggingface_cache'
here = os.getcwd()
os.environ['RETRIV_BASE_PATH'] = here

city_name = 'jacksonville'
se_agenda = sr_agenda = SparseRetriever.load(f'{city_name}_agenda__sparse')
se_transcript = sr_transcript = SparseRetriever.load(f'{city_name}_transcript__sparse')
dr_transcript = DenseRetriever.load(f'{city_name}_transcript', use_gpu=True, transformers_cache_dir=hf_cache_dir)

# create indices for easier lookup
agenda_index_df = pd.Series(sr_agenda.doc_index.index_keys).to_frame('index_key')
agenda_index_df['quoted_url'] = agenda_index_df['index_key'].str.split('/').str.get(-1).str.split('.schedule').str.get(0)
transcript_index_df = pd.Series(sr_transcript.doc_index.index_keys).to_frame('index_key')
transcript_index_df['quoted_url'] = transcript_index_df['index_key'].str.split('___').str.get(0)

meeting_policy_df = pd.read_csv(f'../data/{city_name}__policy_info.csv')
meeting_df = pd.read_csv(f'../data/{city_name}__meeting_info.csv')
meeting_df['Meeting Date'] = pd.to_datetime(meeting_df['Meeting Date'])
policy_df = meeting_policy_df.merge(
    meeting_df[['Meeting Details_href', 'Meeting Date' ]], 
    left_on='key', 
    right_on='Meeting Details_href'
)

# get all agenda items discussed within a week of the policy meeting
target_meeting = policy_df.iloc[0]
target_key_str = quote(target_meeting['video_url'], safe='')
target_schedule_items = list(filter(lambda x: target_key_str in x, sr_agenda.doc_index.index.keys()))
target_transcript_items = list(filter(lambda x: target_key_str in x, sr_transcript.doc_index.index.keys()))

# get recent meetings
start_time = (target_meeting['Meeting Date'] - datetime.timedelta(days=7)) 
end_time = (target_meeting['Meeting Date'] - datetime.timedelta(days=0))
recent_meetings = policy_df.loc[lambda df: df['Meeting Date'].between(start_time, end_time)]
meeting_filter_strings = recent_meetings['video_url'].dropna().drop_duplicates().apply(lambda x: quote(x, safe=''))
recent_schedule_items = get_relevant_indices(meeting_filter_strings, agenda_index_df)
recent_transcript_items = get_relevant_indices(meeting_filter_strings, transcript_index_df)

# get policy discussion by meeting
policy_to_search = target_meeting['Title']
exact_agenda_results = sr_agenda.search(policy_to_search, include_id_list=target_schedule_items)
exact_transcript_results = sr_transcript.search(policy_to_search, include_id_list=target_transcript_items)

# get related policy discussion in recent meetings
time_window_agenda_results = sr_agenda.search(policy_to_search, include_id_list=recent_schedule_items)
time_window_transcript_results = sr_transcript.search(policy_to_search, include_id_list=recent_transcript_items)

# get related policy discussion in all prior meetings
end_time = (target_meeting['Meeting Date'] - datetime.timedelta(days=1))
all_prior_schedule_items, all_prior_transcript_items = get_all_ids_by_start_and_end_dates(
    policy_df, end_date=end_time, agenda_index_df=agenda_index_df, transcript_index_df=transcript_index_df
)

all_agenda_results = sr_agenda.search(policy_to_search, include_id_list=all_prior_schedule_items, cutoff=10)
all_transcript_results = sr_transcript.search(policy_to_search, include_id_list=all_prior_transcript_items, cutoff=10)


############################################################################################################
python retriv_match_files.py  \
    --city_name durham \
    --num_rows 20 \
    --policy_text_output_file durham-policies-matched-with-speakers-test.jsonl \
    --match_policies_to_transcripts \
    --match_policies_to_news_articles \
    --policy_and_news_article_output_file durham-policies-matched-with-news-articles-test.jsonl

#### old 
video_idx = 21
video_url = meeting_policy_df.dropna().drop_duplicates().dropna().sample().iloc[0]['video_url'] # .iloc[video_idx]
quoted_url = quote(video_url, safe='')
transcribed_video_csv_path = f'../data/{city_name}_transcribed_files/{quoted_url}.transcribed.json'
schedule_csv_path = f'../data/{city_name}_videos/{quoted_url}.schedule.csv'
schedule_df = pd.read_csv(schedule_csv_path, index_col=0)

"""
