import pandas as pd
import argparse

def setup():
    city_council_data_df = pd.read_json('../../data/full_newsworthiness_training_data.jsonl',  lines=True)
    final_matching_df = pd.read_csv('../../data/final-matching-articles-and-meetings.csv', index_col=0)
    # full article information/ text
    # the original final_matching_df doesn't have the full article text, 
    # so you might want to look at the actual text

    json_file = '../../data/sfchron-fetched-articles.jsonl/sfchron-fetched-articles.jsonl'
    articles = []
    import json
    for line in open(json_file, encoding="utf8"):
        articles.append(json.loads(line))

    sf_articles_df = pd.DataFrame(articles)
    final_matching_df['key'] = (final_matching_df['article_url']
        .str.split(')')
        .str.get(-1)
        .str.replace('https://', 'http://')
        .str.replace('www.', '')
        .str.replace('http://sfchronicle.com', '')
    )
    matching_df_with_full_text = (
        sf_articles_df
            .assign(key=lambda df: df['article_url'].str.split('sfchronicle.com').str.get(-1))
            [['key', 'article_text']]
            .merge(final_matching_df, on='key', how='right')
    )

    # Merging policy text with true/false label
    renamed_article_matched_df = matching_df_with_full_text.rename(columns={
        'meeting text': 'policy text',
        'summary_text': 'article summary text',
        'article_text': 'article full text'
    })
    renamed_city_council_data_df = city_council_data_df.rename(columns={
        'text': 'policy text',
        'transcribed_text': 'meeting transcribed text'
    })
    full_merged_df = (
        renamed_article_matched_df[['File #', 'article full text', 'article summary text']]
            .merge(
                right=renamed_city_council_data_df[['proposal_number', 'policy text', 'meeting transcribed text', 'label']], 
                left_on='File #',
                right_on='proposal_number', 
                how='right'
            )
    ).drop(columns='File #')
    return full_merged_df

def create_sample(full_merged_df, num_samples):
    only_policy_and_label_df = full_merged_df[["policy text", "label"]].rename(columns={'policy text' : 'text'})

    # Split data into only True and only False texts
    only_true_policy_df = only_policy_and_label_df.loc[only_policy_and_label_df['label']==True]
    only_false_policy_df = only_policy_and_label_df.loc[only_policy_and_label_df['label']==False]

    # Get num_samples from each df and combine them
    only_true_sample_df = only_true_policy_df.sample(num_samples / 2)
    only_false_sample_df = only_false_policy_df.sample(num_samples / 2)
    text_and_label_sample_df = pd.concat([only_true_sample_df, only_false_sample_df], axis=0)

    # Save as jsonl file
    text_and_label_json = text_and_label_sample_df.to_json('./data/input/sf_text_and_label_sample.jsonl', orient='records', lines=True)

if __name__ == "__main__":
    full_merged_df = setup()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="number of texts to assign concepts to"
    )
    args = parser.parse_args()
    num_samples = (args.num_samples)
    create_sample(full_merged_df, num_samples)