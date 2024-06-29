import jsonlines
from tqdm.auto import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()
    with jsonlines.open(args.input_file, 'r') as f:
        with jsonlines.open(args.output_file, 'w') as out:
            for line in tqdm(f.iter(skip_invalid=True, skip_empty=True)):
                if 'html' in line:
                    line.pop('html')

                if 'article_html' in line:
                    line.pop('article_html')
                out.write(line)