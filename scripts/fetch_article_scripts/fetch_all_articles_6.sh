python cc_scraper.py \
	--input-file philly-inquirer-cc-articles-to-fetch.txt.gz \
	--output-file philly-inquirer-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin

python cc_scraper.py \
	--input-file star-ledger-cc-articles-to-fetch.txt.gz \
	--output-file star-ledger-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin				

python cc_scraper.py \
	--input-file seattle-times-cc-articles-to-fetch.txt.gz \
	--output-file seattle-times-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin

python cc_scraper.py \
	--input-file alexandria-times-cc-articles-to-fetch.txt.gz \
	--output-file alexandria-times-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin	

python cc_scraper.py \
	--input-file columbus-dispatch-cc-articles-to-fetch.txt.gz \
	--output-file columbus-dispatch-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin	

python cc_scraper.py \
	--input-file azcentral-cc-articles-to-fetch.txt.gz \
	--output-file azcentral-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin	

python cc_scraper.py \
	--input-file dallas-morning-news-cc-articles-to-fetch.txt.gz \
	--output-file dallas-morning-news-articles.jsonl \
	--num-concurrent-workers 20 \
	--url-selection-style round-robin	