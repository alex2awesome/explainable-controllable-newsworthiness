import aiohttp
import asyncio
import datetime
import orjson as json
import pandas as pd
from random import choice
from urllib.parse import urlparse, urljoin
import gzip
import xopen
from more_itertools import unique_everseen
import re
import jsonlines
from json.decoder import JSONDecodeError
import argparse
import os
from bs4 import BeautifulSoup
import itertools
from tqdm.asyncio import tqdm
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s')


CONTAINER_URLS = [
    "https://common-crawl-scrape-v2-ukvxfz3sya-wl.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-wn.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-ey.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-ul.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-uw.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-nw.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-rj.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-pd.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-wm.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-ew.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-nn.a.run.app",
    "https://common-crawl-scrape-v2-ukvxfz3sya-uk.a.run.app",
]

get_key = lambda x: re.sub(r'\{.*?\}', '', x).strip()

def get_json_body(x: str) -> Dict[str, Any]:
    try:
        to_return = re.findall(r'\{.*?\}', x)[-1]
        to_return = '{' + to_return.split('{')[-1]
        to_return = json.loads(to_return)
    except Exception as e:
        logging.error(f'Error parsing JSON body: {e} - Line: {x}')
        to_return = {}
    return to_return

def make_output_data(row: Dict[str, Any]) -> Dict[str, Any]:
    output_packet = row['json_packet']
    output_packet['article_url'] = row['url'].split('?')[0]
    output_packet['scrape_timestamp'] = row.get('timestamp', datetime.datetime.now().isoformat())
    return output_packet

def filter_article_df(article_df: pd.DataFrame, fn: str) -> pd.DataFrame:
    already_fetched_urls = []
    with xopen.xopen(filename=fn) as f_handle:
        for line in f_handle:
            try:
                json_obj = json.loads(line)
                already_fetched_urls.append(json_obj['article_url'])
            except:
                continue

    already_fetched_urls = list(map(clean_url, already_fetched_urls))
    already_fetched_urls = list(map(lambda x: x.split(')')[-1].strip(), already_fetched_urls))
    article_df = article_df.loc[lambda df: ~df['url'].str.split(')').str.get(-1).str.strip().isin(already_fetched_urls)]
    logging.info(f'Filtering articles to those already fetched... {len(article_df)} left.')
    return article_df

def clean_url(to_get_url: str) -> str:
    return urljoin(to_get_url, urlparse(to_get_url).path)


async def fetch(
        session: aiohttp.ClientSession,
        url: str, data: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        retries: int = 3
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.post(url, json=data) as response:

                    try:
                        result = await response.json(content_type=None)
                        return result
                    except (aiohttp.ContentTypeError, JSONDecodeError) as e:
                        text = await response.text()
                        result = get_json_body(text)
                        if result is not None:
                            return result
                        content_type = response.headers.get('Content-Type', '')
                        logging.error(f'Unexpected content type: {content_type} - Response: {text[:100]}')
                        return None
            except aiohttp.ClientError as e:
                logging.error(f'ClientError: {e}')
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                logging.error(f'TimeoutError on attempt {attempt + 1}')
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

def post_process(datum: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.extract_links:
        html = datum['html']
        soup = BeautifulSoup(html, 'html.parser')

        links = soup.find_all('a', href=True)
        links = list(filter(lambda x: x.get_text() != '', links))
        links_obj = list(map(lambda x: {'text': x.get_text(), 'href': x['href']}, links))
        datum['links'] = links_obj

    if args.drop_html:
        datum.pop('html', None)
    
    return datum

def batch(iterable: List[Any], n: int = 1) -> List[List[Any]]:
    l = len(iterable)
    output_batches = []
    for ndx in range(0, l, n):
        output_batches.append(iterable[ndx:min(ndx + n, l)])
    return output_batches

def get_api_url(c: Any, url_selection_style: str) -> str:
    if url_selection_style == 'random':
        return choice(c)
    else:
        return next(c)

async def main(args: argparse.Namespace):
    if args.input_file.endswith('gzip') or args.input_file.endswith('gz'):
        f_handle = gzip.open(args.input_file, 'rb')
    else:
        f_handle = open(args.input_file)

    seen = set([])
    unique_article_lines = []
    for line in f_handle:
        if isinstance(line, bytes):
            line = line.decode()
        if len(line) > 2:
            url = line.split()[0].split('?')[0]
            if (url not in seen):
                if args.url_filter is not None:
                    if args.url_filter not in url:
                        continue
                if args.status_filter is not None:
                    json_body = get_json_body(line)
                    if int(json_body.get('status', 0)) not in args.status_filter:
                        continue
                unique_article_lines.append(line)
                seen.add(url)

    article_df = (
        pd.Series(unique_article_lines)
            .to_frame('data')
            .assign(json_packet=lambda df: df['data'].apply(get_json_body))
            .assign(key=lambda df: df['data'].apply(get_key).str.split())
            .loc[lambda df: df['key'].str.len() == 2]
            .assign(url=lambda df: df['key'].str.get(0).apply(clean_url))
            .assign(timestamp=lambda df: df['key'].str.get(1))
            .sample(frac=1)
    )
    logging.info(f'Total articles: {len(article_df)}')

    if args.already_fetched is not None:
        article_df = filter_article_df(article_df, args.already_fetched)
    
    if os.path.exists(args.output_file):
        article_df = filter_article_df(article_df, args.output_file)

    c = CONTAINER_URLS
    if args.url_cycle == 'round-robin':
        c = itertools.cycle(CONTAINER_URLS)

    semaphore = asyncio.Semaphore(args.workers)

    for batch_idx in tqdm(
            range(0, len(article_df) + (2 * args.max_num_articles_cycle), args.max_num_articles_cycle),
            desc='Batches'
    ):
        if os.path.exists(args.output_file):
            article_df = filter_article_df(article_df, args.output_file)

        data = article_df.apply(make_output_data, axis=1).sample(frac=1)
        if len(data) == 0:
            break
        data = data.tolist()
        data = data[:args.max_num_articles_cycle]
        logging.info(f'Fetching articles for batch {batch_idx}...')

        output_data = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for row_to_get in data:
                api_url = get_api_url(c, args.url_cycle)
                tasks.append(fetch(session, api_url, row_to_get, semaphore))

            responses = [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]

            for resp_json in responses:
                if resp_json is not None:
                    resp_json = post_process(resp_json, args)
                    output_data.append(resp_json)

        with xopen.xopen(filename=args.output_file, mode='a') as outfile_handle:
            w = jsonlines.Writer(outfile_handle)
            w.write_all(output_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', dest='input_file', type=str)
    parser.add_argument('--output-file', dest='output_file', help="Should be a `.jsonl` file.", type=str)
    parser.add_argument('--num-concurrent-workers', dest='workers', type=int)
    parser.add_argument('--url-filter', default=None, type=str, help='A string to check for in the URL.')
    parser.add_argument('--status-filter', nargs='+', default=None, type=int, help='A list of status codes to filter out.')
    parser.add_argument('--already-fetched-file', dest='already_fetched', type=str, help='List of URLs that we already have from prior runs.')
    parser.add_argument('--url-selection-style', dest='url_cycle', default='round-robin', help='values= ["random", "round-robin"].')
    parser.add_argument('--drop-html', dest='drop_html', action='store_true', help='Drop the HTML from the response.')
    parser.add_argument('--extract-links', dest='extract_links', action='store_true', help='Extract the links from the HTML.')
    parser.add_argument('--max-num-articles-cycle',
                        help="break up the task every X articles...", default=10_000, type=int)

    args = parser.parse_args()
    asyncio.run(main(args))
