import argparse
import os.path

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import unicodedata
import re
import time
from tqdm.auto import tqdm
from urllib.parse import urlparse
import asyncio
import datetime
tqdm.pandas()

async def process_row(td, col_name, row_chunk=None):
    col_name = col_name.replace(':', '')
    text = td.get_text().strip()
    text = unicodedata.normalize("NFKD", text)
    if row_chunk is None:
        row_chunk = {}
    row_chunk[col_name] = text
    a = td.find('a')
    if a and a.attrs.get('href') is not None:
        row_chunk[f"{col_name}_href"] = a.attrs.get('href')
    return row_chunk

async def get_playwright(headless=True):
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    context = await browser.new_context()
    page = await context.new_page()
    return page, context

async def get_all_meeting_tables(
        url,
        playwright_page,
        playwright_context,
        exclude_non_video=False,
        get_video_here=False
):
    await playwright_page.goto(url)

    # change to "all years"
    await playwright_page.click('#ctl00_ContentPlaceHolder1_lstYears_Input')
    await playwright_page.click('#ctl00_ContentPlaceHolder1_lstYears_DropDown > div > ul > li:nth-child(1)')
    page_num_sel = "#ctl00_ContentPlaceHolder1_gridCalendar_ctl00 > thead > tr.rgPager > td > table > tbody > tr > td > div.rgWrap.rgInfoPart"
    await playwright_page.wait_for_selector(page_num_sel)
    html = await playwright_page.content()
    soup = BeautifulSoup(html, 'lxml')
    page_bar = soup.find(attrs={'class': 'rgWrap rgInfoPart'}).get_text().strip()
    total_num_pages = int(re.search(r'Page \d of (\d+)', page_bar)[1])

    # click through all the meetings
    pager_sel_path = '''#ctl00_ContentPlaceHolder1_gridCalendar_ctl00 > 
                thead > 
                tr.rgPager > 
                td > table > tbody > tr > td > div.rgWrap.rgNumPart > a'''

    all_rows = []
    for i in tqdm(range(1, total_num_pages + 1), desc='collecting meeting pages...'):
        num_pages_sel = f'''{pager_sel_path}:has-text("{i}")'''
        await playwright_page.wait_for_selector(pager_sel_path)
        on_page = await playwright_page.is_visible(num_pages_sel)
        if on_page:
            await playwright_page.locator(num_pages_sel).first.click()
        else:
            print(f'didn\'t find locator at: {i}')
            await playwright_page.locator(f'''{pager_sel_path}[title="Next Pages"]''').first.click()
        time.sleep(5)

        ## parse html
        html = await playwright_page.content()
        soup = BeautifulSoup(html, 'lxml')

        tables = soup.find_all('table', attrs={'class': 'rgMasterTable'})
        for table in tables:
            table_head = table.find('thead')
            table_head = list(filter(lambda x: (x != '\n') and (x.attrs.get('class', [None])[0] != 'rgPager'),
                                     list(table_head.children)))[0]
            table_col_names = list(map(lambda x: x.get_text().strip(), table_head.find_all('th')))
            rows = table.find_all('tr', attrs={'class': ['rgRow', 'rgAltRow']})
            for row in rows:
                td_cols = row.find_all('td')
                row_chunk = {}
                for td_name, td in zip(table_col_names, td_cols):
                    row_chunk = await process_row(td, td_name, row_chunk)
                if get_video_here:
                    video_cell = row.find(attrs={'class': 'videolink'})
                    if video_cell:
                        if 'videoFileNotAvailableLink' not in row.find(attrs={'class': 'videolink'}).attrs['class']:
                            vid_link = row.find(attrs={'class': 'videolink'})
                            video_url = await get_video_link(
                                '#' + vid_link.attrs['id'],
                                playwright_page,
                                playwright_context,
                                url,
                                vid_link.attrs['id']
                            )
                            row_chunk['video_url'] = video_url

                all_rows.append(row_chunk)


    all_rows_df = pd.DataFrame(sorted(all_rows, key=lambda x: -len(x)))

    # filter out future meetings
    date_col = 'Date' if 'Date' in all_rows_df.columns else 'Meeting Date'
    all_rows_df[date_col] = pd.to_datetime(all_rows_df[date_col])
    all_rows_df = all_rows_df.loc[lambda df: df[date_col] < datetime.datetime.now()]
    if exclude_non_video:
        all_rows_df = all_rows_df.loc[lambda df: df['Video_href'].notnull()]

    return all_rows_df


async def get_video_link(locator_id, playwright_page, playwright_context, domain, meeting_page=''):
    page_promise = playwright_context.wait_for_event('page', timeout=10000)
    await playwright_page.locator(locator_id).click()
    try:
        new_page = await page_promise
        await new_page.wait_for_load_state()
        video_url = new_page.url
        await new_page.close()
    except PlaywrightTimeoutError as t:
        print(f'timeout error: {domain}/{meeting_page}')
        video_url = None
    return video_url


async def get_policies_for_all_meetings(
        all_rows_df,
        url,
        playwright_page,
        playwright_context,
        wait_for_content=True,
        get_video_here=True, # if True, get video link from these pages, else get it in get_all_meeting_tables
):
    meeting_detail_hrefs = ['Meeting Details_href', 'Details_href']
    meeting_href = None
    for c in meeting_detail_hrefs:
        if c in all_rows_df.columns:
            meeting_href = c

    meeting_details = (
        all_rows_df
        [meeting_href]
        .dropna()
    )
    url_parts = urlparse(url)
    domain = url_parts.scheme + '://' + url_parts.netloc
    rows_for_all_meetings = []
    for meeting_page in tqdm(meeting_details):
        for _ in range(3):
            try:
                await playwright_page.goto(domain + '/' + meeting_page.strip())
                await playwright_page.wait_for_selector('#ctl00_ContentPlaceHolder1_gridMain_ctl00')
                time.sleep(1)
                ## parse table
                html = await playwright_page.content()
                soup = BeautifulSoup(html, 'lxml')
                tables = soup.find_all('table', attrs={'class': 'rgMasterTable'})
                all_meeting_rows = []
                for table in tables:
                    t_head = table.find('thead')
                    t_body = table.find('tbody')
                    col_names = list(map(lambda x: x.get_text(), t_head.find_all('th')))
                    col_names = list(map(lambda x: unicodedata.normalize("NFKD", x).replace(':', ''), col_names))
                    rows = t_body.find_all('tr')
                    for tr in rows:
                        row_chunk = {}
                        for td_name, td in zip(col_names, tr.find_all('td')):
                            row_chunk = await process_row(td, td_name, row_chunk)
                        all_meeting_rows.append(row_chunk)
                all_meeting_rows_df = pd.DataFrame(all_meeting_rows)

                # get video link
                # if wait_for_content:
                if get_video_here:
                    video_url = get_video_link(
                        '#ctl00_ContentPlaceHolder1_hypVideo',
                        playwright_page,
                        playwright_context,
                        domain,
                        meeting_page
                    )
                    all_meeting_rows_df['video_url'] = video_url

                # append
                all_meeting_rows_df['key'] = meeting_page
                rows_for_all_meetings.append(all_meeting_rows_df)
                break
            except Exception as e:
                print(f'error: {e}')
                print(f'general error on: {domain}/{meeting_page}')
            meeting_page = meeting_page.replace('&Options=info|&Search=', '')
        time.sleep(3)

    final_all_rows_meeting_df = pd.concat(rows_for_all_meetings)
    return final_all_rows_meeting_df

async def main(input_domain, output_file_meeting_info, output_file_policy_info, headless=True):
    if 'alexandria' in input_domain:
        get_video_in_meeting_page = True
    else:
        get_video_in_meeting_page = False

    url = f'https://{input_domain}.legistar.com/Calendar.aspx'
    playwright_page, playwright_context = await get_playwright(headless=headless)
    if os.path.exists(output_file_meeting_info):
        meeting_tables = pd.read_csv(output_file_meeting_info, index_col=0)
    else:
        meeting_tables = await get_all_meeting_tables(url, playwright_page, playwright_context, get_video_here=get_video_in_meeting_page)
        meeting_tables.to_csv(output_file_meeting_info, index=False)
    if not os.path.exists(output_file_policy_info):
        policy_info_df = await get_policies_for_all_meetings(meeting_tables, url, playwright_page, playwright_context, get_video_here=not get_video_in_meeting_page)
        policy_info_df.to_csv(output_file_policy_info, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape city council meeting information.')
    parser.add_argument('--input_domain', type=str, help='Domain to scrape for meeting information.')
    parser.add_argument('--output_file_meeting_info', default=None, type=str, help='File to save meeting information.')
    parser.add_argument('--output_file_policy_info', default=None, type=str, help='File to save policy information.')
    parser.add_argument('--not_headless', action='store_true', help='Run browser in not-headless mode.')
    args = parser.parse_args()

    if args.output_file_meeting_info is None:
        args.output_file_meeting_info = f'{args.input_domain}__meeting_info.csv'

    if args.output_file_policy_info is None:
        args.output_file_policy_info = f'{args.input_domain}__policy_info.csv'

    print(f'input domain: {args.input_domain}')
    asyncio.run(main(args.input_domain,
                     args.output_file_meeting_info,
                     args.output_file_policy_info,
                    not args.not_headless
                     ))




"""
conda install conda-forge::opus-tools
conda install conda-forge::libopus
conda install conda-forge::libevent
conda install anaconda::libvpx
conda install -c conda-forge gstreamer gst-plugins-base gst-plugins-good
conda install -c conda-forge harfbuzz
conda install -c conda-forge enchant 
conda install -c conda-forge libsecret
conda install -c conda-forge flite
conda install -c conda-forge libhyphen
conda install conda-forge::mesa


"""