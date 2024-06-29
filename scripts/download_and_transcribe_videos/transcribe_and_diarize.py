import pandas as pd
import whisperx
import gc
import argparse
from whisperx import DiarizationPipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from typing import Optional, Union
import json
import xopen
import torch
import subprocess
import glob
import os, re
from tqdm.auto import tqdm
import random

transcription_batch_size = 32  # reduce if low on GPU mem
embedding_batch_size = 32  # reduce if low on GPU mem
segmentation_batch_size = 32  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
backup_compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)

class MyDiarizationPipeline(DiarizationPipeline):
    def __init__(
        self,
        model_name=None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        top_level_params = {
            'clustering': 'AgglomerativeClustering',
            'embedding': 'speechbrain/spkrec-ecapa-voxceleb',
            'embedding_batch_size': embedding_batch_size,
            'embedding_exclude_overlap': True,
            # 'segmentation': 'pyannote/segmentation@2022.07',
            'segmentation': 'pyannote/segmentation-3.0',
            'segmentation_batch_size': segmentation_batch_size,
            'use_auth_token': use_auth_token,
            "segmentation_step": 0.3,
        }
        low_level_params = {
            'clustering':
                {'method': 'centroid', 'min_cluster_size': 10, 'threshold': 0.7153814381597874},
            'segmentation':
                {'min_duration_off': 0.5817029604921046}  # , 'threshold': 0.4442333667381752}
        }
        self.model = SpeakerDiarization(**top_level_params)
        self.model.instantiate(low_level_params)
        self.model = self.model.to(device)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the directory containing audio files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the directory to save transcriptions")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference")
    parser.add_argument('--delete_audio_files', action='store_true', help='Delete audio files after processing')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    HF_TOKEN = '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'
    diarize_model = MyDiarizationPipeline(use_auth_token=HF_TOKEN, device=args.device)
    try:
        whisper_model = whisperx.load_model("large-v2", args.device, compute_type=compute_type)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        print("Trying to load the model with backup compute type...")
        whisper_model = whisperx.load_model("large-v2", args.device, compute_type=backup_compute_type)
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=args.device)

    audio_files = glob.glob(os.path.join(args.input_dir, '*.mp3'))
    random.shuffle(audio_files)
    for audio_file in tqdm(audio_files, desc='Transcribing and diarizing'):
        print(f"Processing {audio_file}...")
        audio_filename = os.path.basename(audio_file)
        output_filename = os.path.splitext(audio_filename)[0] + '.transcribed.json'
        output_filepath = os.path.join(args.output_dir, output_filename)
        if os.path.exists(output_filepath):
            print(f"File already exists: {output_filepath}")
            continue

        for _ in range(3):
            try:
                # 1. Transcribe with original whisper (batched)
                audio = whisperx.load_audio(audio_file)
                result = whisper_model.transcribe(audio, batch_size=transcription_batch_size, language='en', task='transcribe')

                # 2. Align whisper output
                result = whisperx.align(
                    result["segments"], alignment_model, metadata, audio, args.device, return_char_alignments=False)

                # 3. Assign speaker labels
                diarize_segments = diarize_model(audio_file)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                segs = result['segments']
                list(map(lambda x: x.pop('words'), segs))

                with xopen.xopen(output_filepath, "w") as f:
                    f.write(json.dumps(result))

                if args.delete_audio_files:
                    os.remove(audio_file)
                break
            except Exception as e:
                print(f"An error occurred while processing {audio_file}: {e}")
                transcription_batch_size = max(1, transcription_batch_size // 2)
                pass

