python cc_scraper.py \
	--input-file florida-times-union-cc-articles-to-fetch.txt.gz \
	--output-file florida-times-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin

python cc_scraper.py \
	--input-file fort-worth-star-telegram-cc-articles-to-fetch.txt.gz \
	--output-file fort-worth-star-telegram-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin

python cc_scraper.py \
	--input-file jax-daily-record-cc-articles-to-fetch.txt.gz \
	--output-file jax-daily-record-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin

python cc_scraper.py \
	--input-file denver-post-cc-articles-to-fetch.txt.gz \
	--output-file denver-post-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin

python cc_scraper.py \
	--input-file durham-herald-sun-cc-articles-to-fetch.txt.gz \
	--output-file durham-herald-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin