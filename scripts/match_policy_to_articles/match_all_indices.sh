#!/bin/bash

cities=("denver" "jacksonville" "durham" "seattle" "newark" "fortworth")

cities=("denver" "durham" "seattle" "newark" "fortworth")
city="jacksonville"
for city in "${cities[@]}"; do
    sbatch match_one_index_endeavour.sh \
        --city_name "$city" \
        --policy_text_output_file "${city}-policies-matched-with-speakers.jsonl" \
        --policy_and_news_article_output_file "${city}-policies-matched-with-news-articles.jsonl" \
        --match_policies_to_transcripts \
        --match_policies_to_news_articles
done