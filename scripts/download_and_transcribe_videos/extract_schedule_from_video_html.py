import os
import urllib.parse
from bs4 import BeautifulSoup
import glob
from tqdm.auto import tqdm
import pandas as pd


lookup_dict = []

for d in [
    'fortworthgov.granicus.com',
    'durhamcounty.granicus.com',
    'denver.granicus.com',
    'jaxcityc.granicus.com',
]:
    lookup_dict.append({
        'domain': d,
        'first_level': '#index',
        'second_level': '.index-point',
        'attr': 'time'
    })

lookup_dict.append({
    'domain': 'seattlechannel.org',
    'first_level': '.videoIndex',
    'second_level': '.seekItem',
    'attr': 'data-seek',
})


lookup_dict.append({
    'domain': 'newark.granicus.com',
    'first_level': '.indexPoints',
    'second_level': 'a',
    'attr': 'time',
})

lookup_df = pd.DataFrame(lookup_dict).set_index('domain')


# python extract_schedule_from_video_html.py --input_file_pattern "*_videos/*.html"
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_pattern', type=str, required=True)

    args = parser.parse_args()
    error_files = []

    files_to_check = glob.glob(args.input_file_pattern)
    for f in tqdm(files_to_check):
        output_fname = f.replace('.html', '.schedule.csv')
        f_i = os.path.basename(f)
        f_p = urllib.parse.unquote(f_i)

        if os.path.exists(output_fname):
            continue
        try:
            d = urllib.parse.urlparse(f_p).netloc.replace('www.', '')
            f_level, s_level, attr = lookup_df.loc[d]
            sel = f'{f_level} > {s_level}'
            with open(f) as f_open:
                soup = BeautifulSoup(f_open.read(), 'lxml')

            index_points = soup.select(sel)

            schedule = []
            for i in index_points:
                schedule.append({
                    'time': i.attrs[attr],
                    'title': i.get_text().strip()
                })

            if len(schedule) == 0:
                error_files.append(f)

            pd.DataFrame(schedule).to_csv(output_fname)
        except Exception as e:
            print(f"An error occurred while processing {f}: {e}")
            print(f"Unquote url: {f_p}")
            error_files.append(f)

    with open('error_files.txt', 'w') as f:
        f.write('\n'.join(error_files))