# jax-daily-record-articles-sans-html.jsonl            20_757
# florida-times-articles-sans-html.jsonl               236_439

# star-ledger-articles-sans-html.jsonl                 946_581
# seattle-times-articles-sans-html.jsonl               985_579
# denver-post-articles-sans-html.jsonl               1_250_819
# fort-worth-star-telegram-articles-sans-html.jsonl    262_342

# raleigh-news-and-observer-articles-sans-html.jsonl   329_394
# durham-herald-articles-sans-html.jsonl               61_481


sbatch create_one_index_endeavour.sh  \
  --index_name jacksonville_index \
  --file_to_index \
    ../data/news_articles/jax-daily-record-articles-sans-html.jsonl \
    ../data/news_articles/florida-times-articles-sans-html.jsonl

########################################################################

sbatch create_one_index_endeavour.sh \
  --index_name newark_index \
  --file_to_index \
    star-ledger-articles-sans-html.jsonl \
  --start_index 0 \
  --end_index 250_000

sbatch create_one_index_endeavour.sh \
  --index_name newark_index \
  --file_to_index \
    star-ledger-articles-sans-html.jsonl \
  --start_index 250_000 \
  --end_index 500_000

sbatch create_one_index_endeavour.sh \
  --index_name newark_index \
  --file_to_index \
    star-ledger-articles-sans-html.jsonl \
  --start_index 500_000 \
  --end_index 750_000

sbatch create_one_index_endeavour.sh \
  --index_name newark_index \
  --file_to_index \
    star-ledger-articles-sans-html.jsonl \
  --start_index 750_000 \
  --end_index 1_000_000

########################################################################

sbatch create_one_index_endeavour.sh \
  --index_name seattle_index \
  --file_to_index \
    seattle-times-articles-sans-html.jsonl \
  --start_index 0 \
  --end_index 250_000

sbatch create_one_index_endeavour.sh \
  --index_name seattle_index \
  --file_to_index \
    seattle-times-articles-sans-html.jsonl \
  --start_index 250_000 \
  --end_index 500_000

sbatch create_one_index_endeavour.sh \
  --index_name seattle_index \
  --file_to_index \
    seattle-times-articles-sans-html.jsonl \
  --start_index 500_000 \
  --end_index 750_000

sbatch create_one_index_endeavour.sh \
  --index_name seattle_index \
  --file_to_index \
    seattle-times-articles-sans-html.jsonl \
  --start_index 750_000 \
  --end_index 1_000_000

########################################################################

sbatch create_one_index_endeavour.sh \
  --index_name fortworth_index \
  --file_to_index \
    fort-worth-star-telegram-articles-sans-html.jsonl

########################################################################

sbatch create_one_index_endeavour.sh \
  --index_name durham_index \
  --file_to_index \
    raleigh-news-and-observer-articles-sans-html.jsonl \
    durham-herald-articles-sans-html.jsonl \
  --start_index 0 \
  --end_index 250_000

sbatch create_one_index_endeavour.sh \
  --index_name durham_index \
  --file_to_index \
    raleigh-news-and-observer-articles-sans-html.jsonl \
    durham-herald-articles-sans-html.jsonl \
  --start_index 250_000 \
  --end_index 500_000

########################################################################

sbatch create_one_index_endeavour.sh \
  --index_name denver_index \
  --file_to_index \
    denver-post-articles-sans-html.jsonl \
  --start_index 0 \
  --end_index 250_000

sbatch create_one_index_endeavour.sh \
  --index_name denver_index \
  --file_to_index \
    denver-post-articles-sans-html.jsonl \
  --start_index 250_000 \
  --end_index 500_000

sbatch create_one_index_endeavour.sh \
  --index_name denver_index \
  --file_to_index \
    denver-post-articles-sans-html.jsonl \
  --start_index 500_000 \
  --end_index 750_000

sbatch create_one_index_endeavour.sh \
  --index_name denver_index \
  --file_to_index \
    denver-post-articles-sans-html.jsonl \
  --start_index 750_000 \
  --end_index 1_000_000

sbatch create_one_index_endeavour.sh \
  --index_name denver_index \
  --file_to_index \
    denver-post-articles-sans-html.jsonl \
  --start_index 1_000_000 \
  --end_index 1_300_000