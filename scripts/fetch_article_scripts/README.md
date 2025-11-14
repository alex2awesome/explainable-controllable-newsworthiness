# Fetch Article & Meeting Records

This folder gathers everything needed to assemble text corpora: it scrapes Legistar for meeting metadata, queries Common Crawl (via hosted extractors) for publisher articles, and cleans the payloads so downstream retrieval is lightweight.

## High-Level Flow
```
Legistar endpoints ─┐        Common Crawl URL manifests ─┐
                    │                                   │
                    ▼                                   ▼
   legistar_scraper_[playwright|selenium].py      cc_scraper.py
           │                                         │
           ▼                                         ▼
  *_meeting_info.csv / *_policy_info.csv   <publisher>-articles.jsonl(.gz)
           │                                         │
           ├─► get_wayback_urls.py (fill gaps)       ├─► delete_html_from_files.py (shrink files)
           │                                         │
           └─► fetch_all_articles_*.sh orchestrate multiple publishers at scale
```

## Script Details
| Script | Role | Notable arguments / behavior |
| --- | --- | --- |
| `legistar_scraper_playwright.py`, `legistar_scraper_selenium.py` | Browser automations that page through Legistar committee calendars and agendas when CSV exports are not available programmatically. Two engines exist to work around site-specific anti-bot measures. | Accept city-specific URLs and optional date filters. Choose Playwright when modern JS routing is required; fall back to Selenium for legacy stacks. |
| `cc_scraper.py` | Core asynchronous scraper. Reads Common Crawl “articles-to-fetch” manifests, posts them to managed extraction services (`CONTAINER_URLS`), deduplicates URLs, optionally drops HTML, and enriches with metadata (authors, publish timestamp, links). | Key flags: `--input-file`, `--output-file`, `--num-concurrent-workers`, `--url-selection-style` (`round-robin` or `random`), `--drop-html`, `--extract-links`, `--status-filter`, `--url-filter`. |
| `cc_old_concurrents_scraper.py` | Compatibility layer for earlier manifest schemas that require slightly different parsing. | Use when the gzip manifests do not match the new regex-friendly format expected by `cc_scraper.py`. |
| `get_wayback_urls.py` | Queries the Internet Archive to backfill article URLs that are missing from Common Crawl or have gone offline. | Feed it a list of seed URLs; it returns the closest archived snapshots. |
| `delete_html_from_files.py` | Post-processor that removes the raw `html` field from existing JSONL gzipped files to reduce disk usage after you have extracted the structured content you need. | Point `--input_dir` to the directory containing JSONL files. |
| `fetch_all_articles_[1-6].sh` | Batch launchers tailored to different sets of publishers/cities. They simply invoke `cc_scraper.py` with the right input manifest, output file, concurrency, and round-robin selection. | Use as reference templates; edit to target your data subset and run in parallel shells for faster ingestion. |

## Inputs & Outputs
- **Inputs**: `*-cc-articles-to-fetch.txt.gz` manifests (WARC offsets + metadata), Legistar URLs, optional publisher-specific filters, optional failure lists for retrying.
- **Outputs**: `*_meeting_info.csv`, `*_policy_info.csv`, `*-meeting-dates.csv`, and `<publisher>-articles[-sans-html].jsonl.gz` ready for indexing.

## Working Notes
- `cc_scraper.py` uses asyncio + aiohttp; tune `--num-concurrent-workers` based on the extractor quota (20 is a safe starting point).
- Set `--drop-html` once you trust the parsed fields; otherwise keep the original HTML for debugging, then run `delete_html_from_files.py` later.
- When Legistar throttles headless browsers, set a longer Playwright timeout or use Selenium with a visible browser to mimic human navigation.
