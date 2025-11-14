# Match Policy to Articles

These scripts connect municipal agenda items/transcripts to the news coverage that referenced them. The toolkit spans dense retrieval index creation, batched matching jobs, and LLM grading for qualitative rationales.

## Overview
```
Policy + meeting CSVs             Article JSONL / transcripts
           │                                   │
           ├──────────────┐                    │
           ▼              ▼                    ▼
 retriv_index_files.py   create_one_index_*.sh (FAISS/Retriv builders)
           │                                   │
           └──────────────┬────────────────────┘
                          ▼
                 match_one_index_endeavour.sh
                          │
         ┌────────────────┴──────────────────────┐
         │                                       │
 policies ↔ transcripts JSONL          policies ↔ news articles JSONL
         │                                       │
         └─────────► query_llms_to_find_policy_articles.py / vanilla_run_vllm.py
                     (LLM verdicts + rationales)
```

## Key Components
| Script | Purpose | Highlights |
| --- | --- | --- |
| `create_one_index_endeavour.sh` | SLURM wrapper that forwards options to `retriv_index_files.py` so you can build one retriv FAISS index per city or modality (articles, transcripts, agendas). | Accepts `--index_name`, `--file_to_index` or `--files_to_index`, optional `--file_pattern_to_index`, `--text_col`, `--id_col`, `--embedding_model`, `--filter_with_lr`, `--filter_by_keywords`. |
| `create_all_indices_filtered.sh`, `create_all_indices_meeting_files.sh`, `create_all_indices_not_filtered.sh`, `create_all_indicies_transcripts.sh` | Batch launchers that call `create_one_index_endeavour.sh` for multiple cities/data types. | Use them as templates for your own city lists. |
| `dense_retriever.py` | Custom retriv encoder that loads modern embedding models (e.g., `nvidia/NV-Embed-v1`, `Salesforce/SFR-Embedding-2_R`), handles pooling quirks, and configures FAISS search params per index. | Provides helper utilities like `make_inverse_index_url` and `get_search_params` so `retriv_match_files.py` can map URLs back to IDs and tune IVF/PQ queries. |
| `retriv_index_files.py` | Python entry point that actually constructs the retriv index from JSONL files or glob patterns, applying filters, chunking, and metadata capture. | Called via the shell scripts above; accepts the same flags plus optional start/end slices. |
| `match_one_index_endeavour.sh` | SLURM launcher for `retriv_match_files.py`. Supports toggles for matching policies to transcripts, news articles, or both. | Flags: `--city_name`, `--policy_text_output_file`, `--policy_and_news_article_output_file`, `--match_policies_to_transcripts`, `--match_policies_to_news_articles`. |
| `retriv_match_files.py` | Runs the actual retrieval and writes JSONL outputs containing nearest neighbors, similarity scores, and metadata linking agenda items to coverage. | Honors the flags passed from `match_one_index_endeavour.sh`. |
| `match_all_indices.sh` | Loops over every city and submits `match_one_index_endeavour.sh`, producing consistent naming for output files (e.g., `seattle-policies-matched-with-news-articles.jsonl`). | Adjust the `cities=(...)` array to control scope. |
| `query_llms_to_find_policy_articles.py` | Generates prompts summarizing a policy and its top article hits, then records LLM judgements on whether an article truly covers the policy. | Works best once retrieval outputs exist; helps lift precision by asking an LLM to verify the link. |
| `vanilla_run_vllm.py`, `vanilla_run_vllm_inner.sh`, `vanilla_run_vllm_outer.sh` | vLLM-based inference harness. Streams prompt batches through a multi-GPU setup, normalizes responses, and shards outputs (`llm_annotations__info__START_END.txt`). | Configure `--prompt_file`, `--id_col`, `--prompt_col`, `--output_file`, `--batch_size`. Outer/inner shell scripts orchestrate chunked processing on the cluster. |
| `dense_retriever.py` helpers | Expose functions like `get_detailed_instruct` when you need to wrap queries with task descriptions, keeping the retrieval instructions consistent. | Import directly when crafting specialized prompts. |

## Inputs & Outputs
- **Inputs**: `CITY__policy_info.csv`, meeting transcripts (`*.transcribed.json`), agenda schedule CSVs, article JSONL corpora, prompt dataframes for LLM evaluation.
- **Outputs**: Retriv index directories under the configured cache (FAISS index, metadata, embeddings), JSONL matches linking policies to transcripts/articles, LLM annotation files summarizing coverage confidence.

## Tips
- When indexing transcripts or agenda schedule CSVs, set `--text_col title` and `--id_col id` as shown in `create_all_indices_meeting_files.sh` to ensure the retriever knows which fields to embed.
- Use `--filter_with_lr` to apply lightweight logistic regression filtering before indexing; this keeps redundant/low-quality crawl data out of FAISS and speeds up matching.
- The SLURM wrappers assume large-memory GPUs (A40) for dense retrieval/LLM work. Modify `--gres`, `--mem`, and Conda env paths to fit your cluster.

