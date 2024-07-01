import argparse
import os.path
import time
import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from tqdm.auto import tqdm
from urllib.parse import urlparse

def process_row(td, col_name, row_chunk=None):
    col_name = col_name.replace(':', '')
    text = td.get_text().strip()
    text = unicodedata.normalize("NFKD", text)
    if row_chunk is None:
        row_chunk = {}
    row_chunk[col_name] = text
    a = td.find('a')
    if a and a.get('href') is not None:
        row_chunk[f"{col_name}_href"] = a.get('href')
    return row_chunk

def get_driver(headless=True):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    s = Service('/path/to/chromedriver')
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver


from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm.auto import tqdm
import time

def get_all_meeting_tables(url, driver):
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'ctl00_ContentPlaceHolder1_lstYears_Input'))).click()
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '#ctl00_ContentPlaceHolder1_lstYears_DropDown > div > ul > li:nth-child(1)'))).click()

    page_num_sel = "#ctl00_ContentPlaceHolder1_gridCalendar_ctl00 > thead > tr.rgPager > td > table > tbody > tr > td > div.rgWrap.rgInfoPart"
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, page_num_sel)))
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    page_bar = soup.find(attrs={'class': 'rgWrap rgInfoPart'}).get_text().strip()
    total_num_pages = int(re.search(r'Page \d+ of (\d+)', page_bar).group(1))

    pager_sel_path = '#ctl00_ContentPlaceHolder1_gridCalendar_ctl00 > thead > tr.rgPager > td > table > tbody > tr > td > div.rgWrap.rgNumPart > a'

    all_rows = []
    for i in tqdm(range(1, total_num_pages + 1), desc='Collecting meeting pages...'):
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f'a[title="{i}"]'))).click()
        except TimeoutException:
            print(f'Did not find locator at: {i}')
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f'{pager_sel_path}[title="Next Pages"]'))).click()
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        table = soup.find('table', attrs={'class': 'rgMasterTable'})
        table_head = table.find('thead').find_all(lambda x: (x != '\n') and (x.get('class', [None])[0] != 'rgPager'), recursive=False)[0]
        table_col_names = [th.get_text().strip() for th in table_head.find_all('th')]
        rows = table.find_all('tr', attrs={'class': ['rgRow', 'rgAltRow']})
        print(f'num rows: {len(rows)}')
        for row in rows:
            td_cols = row.find_all('td')
            row_chunk = {}
            for td_name, td in zip(table_col_names, td_cols):
                row_chunk = process_row(td, td_name, row_chunk)
            all_rows.append(row_chunk)
    all_rows_df = pd.DataFrame(sorted(all_rows, key=lambda x: -len(x)))
    return all_rows_df


from selenium.common.exceptions import NoSuchElementException

def get_policies_for_all_meetings(all_rows_df, url, driver):
    meeting_detail_hrefs = ['Meeting Details_href', 'Details_href']
    meeting_href = None
    for c in meeting_detail_hrefs:
        if c in all_rows_df.columns:
            meeting_href = c

    meeting_details = all_rows_df[meeting_href].dropna()
    url_parts = urlparse(url)
    domain = url_parts.scheme + '://' + url_parts.netloc

    rows_for_all_meetings = []
    for meeting_page in tqdm(meeting_details):
        driver.get(domain + '/' + meeting_page)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ctl00_ContentPlaceHolder1_gridMain_ctl00')))
        time.sleep(0.1)  # Small delay to ensure data is loaded
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', attrs={'class': 'rgMasterTable'})
        t_head = table.find('thead')
        t_body = table.find('tbody')
        col_names = [unicodedata.normalize("NFKD", th.get_text()).replace(':', '') for th in t_head.find_all('th')]
        rows = t_body.find_all('tr')
        all_meeting_rows = []
        for tr in rows:
            row_chunk = {}
            for td_name, td in zip(col_names, tr.find_all('td')):
                row_chunk = process_row(td, td_name, row_chunk)
            all_meeting_rows.append(row_chunk)
        all_meeting_rows_df = pd.DataFrame(all_meeting_rows)

        # Try to get video link
        video_url = None
        try:
            video_link = driver.find_element(By.ID, 'ctl00_ContentPlaceHolder1_hypVideo')
            driver.execute_script("arguments[0].click();", video_link)
            video_url = driver.current_url  # Assuming the video opens in the same tab
        except NoSuchElementException:
            print(f'Video link not found for: {domain}/{meeting_page}')

        all_meeting_rows_df['key'] = meeting_page
        all_meeting_rows_df['video_url'] = video_url
        rows_for_all_meetings.append(all_meeting_rows_df)

    final_all_rows_meeting_df = pd.concat(rows_for_all_meetings)
    return final_all_rows_meeting_df



def main(input_domain, output_file_meeting_info, output_file_policy_info, headless=True):
    url = f'https://{input_domain}.legistar.com/Calendar.aspx'
    driver = get_driver(headless=headless)
    if os.path.exists(output_file_meeting_info):
        meeting_tables = pd.read_csv(output_file_meeting_info, index_col=0)
    else:
        meeting_tables = get_all_meeting_tables(url, driver)
        meeting_tables.to_csv(output_file_meeting_info, index=False)
    if not os.path.exists(output_file_policy_info):
        policy_info_df = await get_policies_for_all_meetings(meeting_tables, url, driver)
        policy_info_df.to_csv(output_file_policy_info, index=False)

    # Close the driver after tasks are completed
    driver.quit()


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

    print(f'Input domain: {args.input_domain}')
    main(args.input_domain, args.output_file_meeting_info, args.output_file_policy_info, not args.not_headless)
