import re

import requests
import jsonlines
import requests
from requests.exceptions import HTTPError, Timeout, RequestException, JSONDecodeError
import time
import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
)


def robust_query(url, retries=10, backoff_factor=20, timeout=5):
    """
    Makes a robust HTTP GET request to a URL, with retries and error handling.

    :param url: The URL to query.
    :param retries: Number of times to retry the request in case of failure.
    :param backoff_factor: A backoff factor to apply between attempts.
    :param timeout: Timeout for the request in seconds.
    :return: JSON response if the request is successful and returns JSON, None otherwise.
    """
    headers = {
        'User-Agent': 'ImAResearcher/1.0 (spangher@usc.edu)',
        'Accept': 'application/json'
    }
    url = re.sub(r'\s+', '', url)
    for attempt in range(retries):
        try:
            logging.info(f"Attempt {attempt + 1} of {retries} for URL: {url}")
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses
            try:
                return response.json()
            except JSONDecodeError:
                logging.error("Response is not in JSON format.")
                return None
        except HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except Timeout as timeout_err:
            logging.error(f"Request timed out: {timeout_err}")
        except RequestException as req_err:
            logging.error(f"Request exception occurred: {req_err}")
        time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
    logging.error(f"Failed to retrieve URL after {retries} attempts")
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()
    START_URL = start_url = f'''http://web.archive.org/cdx/search/cdx?
        url={args.domain}/*
        &limit=10000
        &showResumeKey=true
        &output=json
        &filter=statuscode:200
    '''
    w = jsonlines.open(args.output_file, 'w')


    response = robust_query(START_URL)
    if response is None:
        raise ValueError(f"Failed to retrieve URL: {START_URL}")

    resume_key = response[-1]
    data_output = response[:-2]
    w.write_all(data_output)

    while True:
        NEXT_URL = f'''http://web.archive.org/cdx/search/cdx?
            url={args.domain}/*
            &limit=10000
            &showResumeKey=true
            &output=json
            &filter=statuscode:200
            &resumeKey={resume_key[0]}
        '''
        response = robust_query(NEXT_URL)
        if response is None:
            break
        resume_key = response[-1]
        data_output = response[1:-2]
        w.write_all(data_output)
        time.sleep(20)