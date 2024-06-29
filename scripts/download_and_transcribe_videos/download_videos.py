# conda install -c conda-forge libmagic
# conda install -c conda-forge python-magic

import argparse
import asyncio
import subprocess
import time

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import ffmpeg
from bs4 import BeautifulSoup
import wget
import urllib
import os
from tqdm.asyncio import tqdm
import requests
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s')


# debugging options
QUIET_FFMPEG = True
PARTIAL_DOWNLOAD = False
HEADLESS = True

def check_mimetype(file, first_check=True):
    import magic
    magic = magic.Magic()
    mimetype = magic.from_file(file)
    if 'MP4' in mimetype:
        return 'mp4'
    elif 'MPEG' in mimetype:
        return 'mp3'
    else:
        if first_check:
            subprocess.call(['ffmpeg',  '-y', '-i', file, 'tmp.mp4', '-hide_banner'])
            subprocess.call(['mv', 'tmp.mp4', file])
            return check_mimetype(file, first_check=False)
        logging.info(f"Unknown mimetype: {mimetype}")
        return None


def download_video_ffmpeg_partial(playlist_url, output_file, number_of_segments=30):
    # Download the playlist
    subprocess.call(['ffmpeg',
                     '-i', playlist_url,
                     '-c', 'copy',
                     '-t', str(number_of_segments),
                     '-bsf:a', 'aac_adtstoasc', output_file
                     ])


def download_video_ffmpeg(url, output_file):
    try:
        # Stream input from URL
        input_stream = ffmpeg.input(url)
        # Copy video and audio streams without re-encoding
        output_stream = ffmpeg.output(input_stream, output_file, codec='copy')
        # Run and overwrite output file if exists
        ffmpeg.run(output_stream, overwrite_output=True, quiet=QUIET_FFMPEG)
        logging.info(f"Download complete: {output_file}")
    except Exception as e:
        logging.info("An error occurred:", e)

async def download_video_denver_jacksonville_fortwoth_durham(page, url, output_file_name):
    for _ in range(3):
        if os.path.exists(output_file_name):
            return
        try:
            logging.info(f"Navigating to: {url}")
            await page.goto(url, timeout=30_000)
            logging.info(f"Starting download from: {url}")
            time.sleep(5)
        except PlaywrightTimeoutError:
            logging.info(f"Timeout error: {url}")
            pass
        try:
            download_button_success = False
            download_sel = '#navigation-display-download > span'
            exists_download_button = await page.query_selector(download_sel)
            if exists_download_button:
                logging.info('download button exists, clicking...')
                await page.locator(download_sel).click()
                video_sel = '#download-options > a > span > i.fa.fa-file-video-o.fa-stack-1x.fa-inverse'
                exists_video_download_button = await page.query_selector(video_sel)
                if exists_video_download_button:
                    async with page.expect_download(timeout=0) as download_info:
                        await page.locator(video_sel).click()
                    logging.info('button clicked...')
                    download = await download_info.value
                    await download.save_as(output_file_name)
                    logging.info(f"Download complete: {output_file_name}")
                    download_button_success = True
                else:
                    download_button_success = False
            if not download_button_success:
                logging.info('download button doesn\'t exist')
                video_tag_sel = 'video'
                await page.wait_for_selector(video_tag_sel, timeout=60000)
                html = await page.content()
                soup = BeautifulSoup(html, 'lxml')
                video_tag = soup.find(video_tag_sel).find('source')
                playlist_url = video_tag.attrs['src']
                if PARTIAL_DOWNLOAD:
                    download_video_ffmpeg_partial(playlist_url, output_file_name)
                else:
                    download_video_ffmpeg(playlist_url, output_file_name)
        except Exception as e:
            logging.info(f"An error occurred while downloading: {e}")
            time.sleep(5)


async def download_video_seattle(page , url, output_file_name):
    for _ in range(3):
        try:
            await page.goto(url, timeout=30_000)
            logging.info(f"Starting download from: {url}")
            time.sleep(5)
        except PlaywrightTimeoutError:
            logging.info(f"Timeout error: {url}")
            pass

        try:
            if os.path.exists(output_file_name):
                return
            await page.locator('#vidPlayer').click(timeout=60_000)
            await page.hover('#vidPlayer')
            try:
                await page.locator('''
                    #vidPlayer > 
                        div.jw-wrapper.jw-reset > 
                        div.jw-controls.jw-reset > 
                        div.jw-controlbar.jw-reset > 
                        div.jw-reset.jw-button-container > 
                        div:nth-child(13) > 
                        div.jw-icon.jw-button-image.jw-button-color.jw-reset
                ''').click(timeout=60_000)
            except PlaywrightTimeoutError:
                pass
            page_url = page.url
            wget.download(page_url, out=output_file_name)
        except Exception as e:
            logging.info(f"An error occurred while downloading: {e}")
            time.sleep(5)
            pass

async def download_video_alexandria_newark(page, url, output_filename):
    for _ in range(3):
        try:
            await page.goto(url, timeout=30_000)
            logging.info(f"Starting download from: {url}")
            time.sleep(5)
        except PlaywrightTimeoutError:
            logging.info(f"Timeout error: {url}")
            pass
        try:
            if os.path.exists(output_filename):
                return

            download_sel = '#download-menu'
            await page.locator(download_sel).click()
            # await download_with_progress(page, download_sel, output_filename, url)

            async with page.expect_download(timeout=0) as download_info:
                logging.info(f"Downloading via button from: {url}")
                # Perform the action that initiates download
                download_sel = '#downloadVideo'
                await page.locator(download_sel).click(timeout=0)
            download = await download_info.value
            await download.save_as(output_filename)
        except Exception as e:
            logging.info(f"An error occurred while downloading: {e}")
            time.sleep(5)
            pass


def get_audio_or_video_file_length(file_path):
    proc = subprocess.Popen([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1',
        file_path
    ], stdout=subprocess.PIPE)
    output = proc.stdout.read().decode().strip()
    try:
        return float(output)
    except ValueError:
        logging.info(f"Error converting file length to float: {output}")
        return None


def convert_to_audio_and_delete(video_file_path, audio_file_path):
    is_mp4 = check_mimetype(video_file_path)
    if is_mp4 is None:
        return
    else:
        logging.info(f"Converting {video_file_path} to {audio_file_path}")

    subprocess.call([
        'ffmpeg', '-n', '-i',
        video_file_path,
        '-vn', '-q:a', '0', '-map', 'a',
        audio_file_path
    ])
    video_file_length = get_audio_or_video_file_length(video_file_path)
    audio_file_length = get_audio_or_video_file_length(audio_file_path)
    if video_file_length is not None and audio_file_length is not None:
        logging.info(f"Video file length: {video_file_length} seconds")
        logging.info(f"Audio file length: {audio_file_length} seconds")
        if (video_file_length - audio_file_length) < (.005 * video_file_length):
            os.remove(video_file_path)

async def get_browser(output_dir, not_headless):
    playwright = await async_playwright().start()
    if ('alexandria' in output_dir) or ('newark' in output_dir) or ('fortworth' in output_dir):
        browser = await playwright.chromium.launch(headless=not not_headless)
    else:
        browser = await playwright.firefox.launch(headless=not not_headless)
    context = await browser.new_context()
    page = await context.new_page()
    return page

async def download_all_videos(data_df, output_dir, not_headless, skip_transcription=True):
    for _ in range(3):
        try:
            page = await get_browser(output_dir, not_headless)
            video_urls = (
                data_df['video_url']
                    .loc[lambda s: s.str.len() < 200]
                    .dropna().drop_duplicates()
                    .sample(frac=1).tolist()
            )
            for url in tqdm(video_urls, desc='Downloading videos'):
                output_filename = urllib.parse.quote(url, safe='')
                output_filepath = os.path.join(output_dir, output_filename)
                if (
                        os.path.exists(output_filepath + '.mp3') or
                        # os.path.exists(output_filepath + '.mp4') or
                        (os.path.exists(output_filepath + '.transcribed.json') and skip_transcription)
                ):
                    logging.info(f"File already exists: {output_filepath}")
                    continue

                if 'denver' in output_dir or 'jacksonville' in output_dir or 'fortworth' in output_dir or 'durham' in output_dir:
                    await download_video_denver_jacksonville_fortwoth_durham(page, url, output_filepath + '.mp4')
                elif 'seattle' in output_dir:
                    await download_video_seattle(page, url, output_filepath + '.mp4')
                elif 'alexandria' in output_dir or 'newark' in output_dir:
                    await download_video_alexandria_newark(page, url, output_filepath + '.mp4')

                # convert to audio
                if os.path.exists(output_filepath + '.mp4'):
                    convert_to_audio_and_delete(output_filepath + '.mp4', output_filepath + '.mp3')

                html = await page.content()
                with open(output_filepath + '.html', 'w') as f:
                    f.write(html)

        except Exception as e:
            logging.info(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download videos from a list of URLs provided in a CSV file.")
    parser.add_argument("--input_file", help="Path to the CSV file containing video URLs")
    parser.add_argument("--output_dir", help="Directory to save downloaded videos")
    parser.add_argument('--not_headless', action='store_true', help='Run browser in head mode')
    args = parser.parse_args()
    import pandas as pd
    data_df = pd.read_csv(args.input_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    asyncio.run(download_all_videos(data_df, args.output_dir, args.not_headless))

if __name__ == "__main__":
    main()


# 
#  mkdir sample_htmls_to_download
# import glob
# import os
# import random
# video_folders = glob.glob('*videos')
# html_sample = []
# for v in video_folders:
#     htmls = glob.glob(f'{v}/*.html')
#     random.shuffle(htmls)
#     html_sample += htmls[:5]
#
# for h in html_sample:
#     f = os.path.basename(h)
#     shutil.copy(h, f'sample_htmls_to_download/{f}')
