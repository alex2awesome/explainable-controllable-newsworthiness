# Explainable & Controllable Newsworthiness

This repository assembles city-council news ecosystems so we can study which policy actions become news, generate transparent retrieval pipelines, and condition large language models (LLMs) on factual civic signals. The repo currently focuses on six fully-populated cities (Seattle, Denver, Fort Worth, Jacksonville, Philadelphia, Phoenix) while scaffolding additional municipalities.

## Workflow at a Glance
```
                ┌──────────────────────────────┐
                │ scripts/fetch_article_scripts│
                └──────────────┬───────────────┘
                               │
                 Legistar agendas, meeting rolls,
                  Common Crawl URL manifests
                               │
 ┌──────────────────────────────┴──────────────────────────────┐
 │ City video pages & audio    +    News articles & policy CSVs│
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
         ┌─────────────────────────────────────────┐
         │scripts/download_and_transcribe_videos   │
         └──────────────────┬──────────────────────┘
                            │
                    mp4/audio + WhisperX transcripts
                            │
                            ▼
         ┌─────────────────────────────────────────┐
         │scripts/match_policy_to_articles         │
         └─────────────────────────────────────────┘
                            │
          policy ↔ article matches + LLM rationales
```

## Repository Layout
- `scripts/` – production pipelines (detailed below) for scraping, AV ingestion, transcription, and policy/article linkage.
- `data/` – per-city corpora (meeting metadata, agenda items, URL manifests, scraped article JSONL, raw videos).
- `notebooks/` – ad-hoc EDA and prototype analyses.
- `CB-LLM/`, `topicGPT/` – model-specific experiments and prompts used for controllable topic modeling.

## Scripts
Each sub-folder in `scripts/` holds a self-contained stage of the pipeline. Most scripts accept CLI arguments; inspect the file headers for authoritative options.

### `scripts/download_and_transcribe_videos`
Purpose: automate capture of council meeting videos/audio, recover chapter markers, and run diarized transcriptions.

- `download_videos.py` – Playwright-driven grabber for multiple municipal streaming stacks (Granicus, Seattle Channel, etc.), with ffmpeg fallbacks and MIME detection to normalize MP4/MP3.
- `download_all_videos.sh` / `run_all_files.sh` – batch runners that iterate through per-city URL manifests.
- `sbatch_run_one_transcription.sh` – SLURM helper for GPU queues.
- `transcribe_and_diarize.py` – wraps WhisperX + custom `MyDiarizationPipeline` (pyannote) to produce `*.transcribed.json` aligned segments and speaker tags; optionally prunes intermediate audio.
- `extract_schedule_from_video_html.py` – parses embedded agenda indices from saved HTML surrounding a video player and emits `*.schedule.csv`, relying on domain-specific CSS selectors.

### `scripts/fetch_article_scripts`
Purpose: curate meeting metadata and article corpora from Legistar and Common Crawl, then clean intermediate artifacts.

- `legistar_scraper_playwright.py` / `legistar_scraper_selenium.py` – alternative browser automations for cities whose Legistar instances require scripted navigation.
- `cc_scraper.py` – asynchronous Common Crawl fetcher that posts to hosted scrape endpoints, deduplicates URLs, optionally strips HTML, and emits JSONL with metadata (authors, publish date, links, etc.).
- `cc_old_concurrents_scraper.py` – legacy compatibility layer for prior crawl manifests.
- `fetch_all_articles_[1-6].sh` – thin wrappers that partition URL manifests by city/publisher and queue the scraper with the right filters.
- `get_wayback_urls.py` – uses the Wayback Machine to back-fill missing articles when Common Crawl coverage is sparse.
- `delete_html_from_files.py` – post-processing utility to drop raw HTML blobs once structured fields have been persisted, keeping gzip payloads small.

### `scripts/match_policy_to_articles`
Purpose: surface which agenda items or meetings manifested in the press, by combining dense retrieval, index building, and LLM grading.

- `create_*` / `match_*` shell scripts – orchestrate ingestion of article/policy corpora into retriv FAISS indexes (filtered by modality such as meeting files vs policy transcripts) and run bulk similarity searches.
- `dense_retriever.py` – custom wrapper around retriv encoders (NV-Embed, Salesforce SFR) that builds ANN searchers, maintains inverse URL indexes, and exposes configurable FAISS search params.
- `retriv_index_files.py` / `retriv_match_files.py` – helpers that register new collections and run batched retrieval jobs against article corpora.
- `query_llms_to_find_policy_articles.py` – builds structured prompts to ask LLMs whether a specific policy was covered, allowing manual verification.
- `vanilla_run_vllm.py` with `vanilla_run_vllm_[inner|outer].sh` – vLLM-based inference harness that streams prompts in batches, enforces Hugging Face auth, and writes normalized JSONL responses.

## Data Folder
The `data/` directory is organized by city. Completed cities share the following structure:

- `CITY__meeting_info.csv` – meeting-level records (committee name, date/time, room, agenda/minutes/Video links) exported from Legistar; underscores avoid clashes with publisher names.
- `CITY__policy_info.csv` – agenda-line granularity (File/Record IDs, ordinance titles, action taken, video URL) for the same jurisdiction.
- `CITY-meeting-info.csv` – agenda items re-shaped with additional context (meeting name/date/time appended to each row); present where we have enriched versions.
- `CITY-meeting-dates.csv` – schedule-focused table enumerating every posted meeting plus links to agendas, packets, minutes, results, and eComment portals.
- `<publisher>-cc-articles-to-fetch.txt.gz` – newline-delimited records from Common Crawl (WARC filename, offsets, HTTP status) that seed article scraping.
- `<publisher>-articles-sans-html.jsonl.gz` – cleaned article payloads with `article_text`, `article_url`, authorship, publish timestamp, and scrape metadata. HTML is optionally stripped to control file size.
- Extra assets (when available): raw HTML players, MP4s, or derived `*.schedule.csv` from video downloads (e.g., Jacksonville’s `jacksonville_videos/`).

### Fully Populated City Corpora
| City | Meeting schedule files | Agenda/policy detail | News article assets | Extras |
| --- | --- | --- | --- | --- |
| **Seattle** | `seattle__meeting_info.csv`, `seattle-meeting-dates.csv` | `seattle__policy_info.csv`, `seattle-meeting-info.csv` (agenda rows tied to specific committees) | `seattle-times-cc-articles-to-fetch.txt.gz`, `seattle-times-articles-sans-html.jsonl.gz` | Generated `*.schedule.csv` when video HTML is parsed. |
| **Denver** | `denver__meeting_info.csv` | `denver__policy_info.csv` | `denver-post-cc-articles-to-fetch.txt.gz`, `denver-post-articles-sans-html.jsonl.gz` | — |
| **Fort Worth** | `fortworthgov__meeting_info.csv` | `fortworthgov__policy_info.csv` | `fort-worth-star-telegram-cc-articles-to-fetch.txt.gz`, `fort-worth-star-telegram-articles-sans-html.jsonl.gz` | Links in `__policy_info` already point to Granicus clips for downstream transcription. |
| **Jacksonville** | `jaxcityc__meeting_info.csv` | `jaxcityc__policy_info.csv` | `florida-times-union-cc-articles-to-fetch.txt.gz`, `florida-times-articles-sans-html.jsonl.gz`, `jax-daily-record-cc-articles-to-fetch.txt.gz` | `jacksonville_videos/` stores saved HTML + MP4 assets, enabling local transcription and schedule extraction. |
| **Philadelphia** | `philadelphia-meeting-info.csv`, `philadelphia-meeting-dates.csv` | (agenda rows embedded within `philadelphia-meeting-info.csv`) | `philly-inquirer-cc-articles-to-fetch.txt.gz`, `inquirer-urls-to-get.csv` (curated fetch list) | — |
| **Phoenix** | `phoenix-meeting-info.csv`, `phoenix-meeting-dates.csv` | Agenda items include File #, action type, and meeting metadata in the same CSV | `azcentral-cc-articles-to-fetch.txt.gz` | — |

Additional folders (e.g., `alexandria/`, `columbus/`, `newark/`, `data/full-data/`, `data/gpt_classification/`) hold partial scrapes, aggregation outputs, and model-ready datasets that are still in flight.

### Working with the Data
- All `*.gz` files are standard gzip streams; use `gzip -dc file.gz | head` or Python’s `gzip` module to peek without inflating to disk.
- CSVs originated from Legistar exports, so column names match the platform’s UI; most have `_href` companions pointing to canonical URLs.
- When building retrieval indexes, point the scripts in `scripts/match_policy_to_articles/` to the per-city article JSONL plus the `__policy_info` files to stay aligned with the schema assumed by `dense_retriever.py`.

## Next Steps
1. Use the fetch scripts to expand partially-populated cities (Alexandria, Newark, etc.) so they match the six reference corpora.
2. Wire the WhisperX transcripts and `*_schedule.csv` files into the retrieval stage to let LLMs cite exact timestamps when explaining newsworthiness.
3. Keep README synced as new cities or publishers become “complete”.

