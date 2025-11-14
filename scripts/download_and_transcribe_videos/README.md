# Download & Transcribe Videos

Tools in this folder turn council video links into diarized transcripts that downstream retrieval can cite. The steps are intentionally modular so you can re-run only the pieces you need (download, schedule extraction, transcription, cleanup).

## Pipeline
```
Legistar policy CSVs
   (video_url columns)
         │
         ▼
 download_videos.py  ──►  *.mp4 / *.mp3  ──┐
         │                                 │
         ├─► extract_schedule_from_video_html.py ──► *.schedule.csv (chapter markers)
         │                                 │
         └─► sbatch_run_one_transcription.sh / transcribe_and_diarize.py
                                             │
                                             ▼
                                      *.transcribed.json
                                      (WhisperX + diarization)
```

## Scripts
| Script | What it does | Key flags / notes |
| --- | --- | --- |
| `download_videos.py` | Opens each `video_url` using Playwright, handles Seattle Channel + Granicus variants, and saves audio/video via ffmpeg or direct HTTP download. Detects MIME types with `python-magic`. | `--input_file` expects a policy or meeting CSV with `video_url` column; `--output_dir` is where MP4/MP3 files land. Environment variables `HEADLESS`, `PARTIAL_DOWNLOAD`, `QUIET_FFMPEG` can be toggled at the top of the file. |
| `download_all_videos.sh` | Convenience wrapper that calls `download_videos.py` for multiple cities using their `__policy_info.csv` files. | Edit to match the jurisdictions you need; runs sequentially. |
| `extract_schedule_from_video_html.py` | Parses saved HTML player pages to recover embedded agenda chapters (time + label) and emits `*.schedule.csv`. Domain-specific CSS selectors live inside the script. | `--input_file_pattern "*_videos/*.html"` to target HTML dumps per city. Creates `error_files.txt` with failures for manual follow-up. |
| `transcribe_and_diarize.py` | Loads WhisperX + pyannote diarization (custom `MyDiarizationPipeline`) to produce aligned transcripts with speaker labels. Removes word-level detail to keep files light. | `--input_dir` points to MP3s (convert beforehand if you only have MP4s). `--device` auto-detects CUDA; set `--delete_audio_files` to clean up once transcripts succeed. Adjust global batch sizes if GPUs are memory-constrained. |
| `sbatch_run_one_transcription.sh` | SLURM submission helper. Ensures the input directory exists locally (optionally fetching a tarball from GCS), activates the right Conda env, and launches `transcribe_and_diarize.py`. | Usage: `sbatch sbatch_run_one_transcription.sh seattle_videos seattle_transcribed_files`. |
| `run_all_files.sh` | Example batch queue that submits the transcription job for each city directory. | Edit list to match your storage naming scheme. |

## Inputs & Outputs
- **Inputs**: `*_policy_info.csv` or `*_meeting_info.csv` files (with `video_url` columns), HTML player downloads (optional but required for schedule extraction), GPU access for transcription.
- **Outputs**: video/audio binaries, `*.schedule.csv` (chapter list), `*.transcribed.json` (WhisperX JSON emitted via xopen). These live in per-city folders such as `jacksonville_videos/` and `jacksonville_transcribed_files/`.

## Usage Tips
- When Playwright hits rate limits, toggle `HEADLESS=False` and add manual delays to debug selectors.
- Some Granicus instances lack download buttons. The script automatically falls back to parsing `<video>` tags and streaming via ffmpeg; make sure ffmpeg is on `$PATH`.
- Pyannote models require a Hugging Face token (`HF_TOKEN` inside `transcribe_and_diarize.py`). Keep it valid in your execution environment.

