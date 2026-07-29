"""
batch_soap_analysis.py
======================
Batch processor for the Medical STT evaluation pipeline.

For each audio file:
  1. Transcribes with faster-whisper (GPU float16) — cached if already done
  2. Sends segments to Gemini for SOAP extraction
  3. Appends one row to: all_audio_soap/predicted/all_predicted.csv

For each matching clean transcript:
  1. Reads the .txt ground-truth transcript
  2. Sends raw text to Gemini for SOAP extraction
  3. Appends one row to: all_audio_soap/ground_truth/all_ground_truth.csv

Usage:
  # Full batch (audio + ground truth)
  python batch_soap_analysis.py

  # Audio only (predicted)
  python batch_soap_analysis.py --mode predicted

  # Ground truth only (clean transcripts)
  python batch_soap_analysis.py --mode ground_truth

  # Process a subset
  python batch_soap_analysis.py --limit 10

  # Skip already-processed files (resume interrupted batch)
  python batch_soap_analysis.py --resume
"""

import argparse, json, os, sys, time, traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR   = Path(__file__).parent
AUDIO_DIR    = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Audio_Recordings'
TRANSCRIPT_DIR = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Clean_Transcripts'
OUTPUT_DIR   = SOURCE_DIR / 'all_audio_soap'
PREDICTED_DIR   = OUTPUT_DIR / 'predicted'
GROUND_TRUTH_DIR = OUTPUT_DIR / 'ground_truth'
LOG_DIR      = OUTPUT_DIR / 'logs'

for d in [PREDICTED_DIR, GROUND_TRUTH_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COMBINED_PREDICTED_CSV   = PREDICTED_DIR   / 'all_predicted.csv'
COMBINED_GROUND_TRUTH_CSV = GROUND_TRUTH_DIR / 'all_ground_truth.csv'
ERROR_LOG = LOG_DIR / f'errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

# Delay between Gemini calls (seconds) to avoid 503 rate limits
GEMINI_DELAY = 3

# ---------------------------------------------------------------------------
# Import pipeline (adds LD_LIBRARY_PATH at import time via activate patch,
# but for direct python invocation we set it here as a safety net)
# ---------------------------------------------------------------------------
_CUDA_LIB = (
    SOURCE_DIR.parent /
    'medical-stt-env/lib/python3.14/site-packages/nvidia/cu13/lib'
)
if _CUDA_LIB.exists():
    os.environ['LD_LIBRARY_PATH'] = (
        str(_CUDA_LIB) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    )

sys.path.insert(0, str(SOURCE_DIR))
from stt_pipeline_gpu import (
    transcribe_audio,
    diarize_and_extract_soap,
    diarize_and_extract_soap_from_text,
    _flatten,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_error(audio_name: str, stage: str, exc: Exception):
    msg = f'[{datetime.now().isoformat()}] {audio_name} | stage={stage} | {type(exc).__name__}: {exc}\n'
    print(f'  ⚠️  {msg.strip()}')
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(msg)
        traceback.print_exc(file=f)
        f.write('\n')


def _soap_to_row(result: dict, source_name: str) -> dict | None:
    """Convert Gemini SOAP result dict → flat CSV row dict."""
    if result is None:
        return None
    soap = {k: v for k, v in result.items() if k != 'diarized_segments'}
    flat = pd.json_normalize(soap, sep='.')
    flat = _flatten(flat)
    flat.insert(0, 'audio_file', source_name)
    return flat


def _append_to_csv(row_df: pd.DataFrame, csv_path: Path):
    """Append a single-row DataFrame to a CSV, writing header only on first write."""
    write_header = not csv_path.exists()
    row_df.to_csv(csv_path, mode='a', index=False, header=write_header)


def _already_processed(source_name: str, csv_path: Path) -> bool:
    """Check if this audio file already has a row in the combined CSV."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, usecols=['audio_file'])
        return source_name in df['audio_file'].values
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Batch: predicted (audio → STT → SOAP)
# ---------------------------------------------------------------------------

def run_predicted_batch(audio_files: list[Path], force: bool):
    print(f'\n{"="*60}')
    print(f'  BATCH: Predicted SOAP  ({len(audio_files)} files)')
    print(f'  Output: {COMBINED_PREDICTED_CSV}')
    if force:
        print(f'  Mode: --force (re-process everything, overwrite existing rows)')
    else:
        print(f'  Mode: skip already-processed files (use --force to reprocess)')
    print(f'{"="*60}')

    success, skipped, failed = 0, 0, 0

    for i, audio_path in enumerate(audio_files, 1):
        name = audio_path.name
        print(f'\n[{i}/{len(audio_files)}] {name}')

        if not force and _already_processed(name, COMBINED_PREDICTED_CSV):
            print('  → Already in CSV, skipping (use --force to reprocess)')
            skipped += 1
            continue

        # Stage 1: Transcribe (uses cache if available)
        try:
            segments = transcribe_audio(str(audio_path), use_cache=True)
        except Exception as e:
            _log_error(name, 'transcription', e)
            failed += 1
            continue

        if not segments:
            print(f'  ⚠️  No segments — skipping')
            _log_error(name, 'transcription', ValueError('Empty segments'))
            failed += 1
            continue

        # Stage 2: Gemini SOAP extraction (with retry on 503)
        result = None
        for attempt in range(1, 4):
            try:
                result = diarize_and_extract_soap(segments)
                break
            except Exception as e:
                err_str = str(e)
                if '503' in err_str and attempt < 3:
                    wait = 15 * attempt
                    print(f'  ⚠️  Gemini 503, retrying in {wait}s (attempt {attempt}/3)...')
                    time.sleep(wait)
                else:
                    _log_error(name, 'gemini_soap', e)
                    break

        if result is None:
            failed += 1
            continue

        # Stage 3: Append to combined CSV + save individual file
        row_df = _soap_to_row(result, name)
        if row_df is not None:
            # If force-reprocessing, remove old row from combined CSV first
            if force and COMBINED_PREDICTED_CSV.exists():
                existing = pd.read_csv(COMBINED_PREDICTED_CSV)
                existing = existing[existing['audio_file'] != name]
                existing.to_csv(COMBINED_PREDICTED_CSV, index=False)
            _append_to_csv(row_df, COMBINED_PREDICTED_CSV)
            # Also save / overwrite individual CSV
            indiv_path = PREDICTED_DIR / f'{audio_path.stem}_soap.csv'
            row_df.to_csv(indiv_path, index=False)
            print(f'  ✅ Saved → {indiv_path.name}')
            success += 1
        else:
            _log_error(name, 'save', ValueError('SOAP result was None after extraction'))
            failed += 1

        # Rate-limit Gemini calls
        if i < len(audio_files):
            time.sleep(GEMINI_DELAY)

    print(f'\n{"="*60}')
    print(f'  Predicted batch done: {success} ✅  {skipped} skipped  {failed} ❌')
    print(f'  Combined CSV: {COMBINED_PREDICTED_CSV}')
    print(f'{"="*60}')
    return success, skipped, failed

# ---------------------------------------------------------------------------
# Batch: ground truth (clean transcript → SOAP)
# ---------------------------------------------------------------------------

def run_ground_truth_batch(transcript_files: list[Path], force: bool):
    print(f'\n{"="*60}')
    print(f'  BATCH: Ground Truth SOAP  ({len(transcript_files)} files)')
    print(f'  Output: {COMBINED_GROUND_TRUTH_CSV}')
    if force:
        print(f'  Mode: --force (re-process everything, overwrite existing rows)')
    else:
        print(f'  Mode: skip already-processed files (use --force to reprocess)')
    print(f'{"="*60}')

    success, skipped, failed = 0, 0, 0

    for i, txt_path in enumerate(transcript_files, 1):
        name = txt_path.name
        print(f'\n[{i}/{len(transcript_files)}] {name}')

        if not force and _already_processed(name, COMBINED_GROUND_TRUTH_CSV):
            print('  → Already in CSV, skipping (use --force to reprocess)')
            skipped += 1
            continue

        try:
            raw_text = txt_path.read_text(encoding='utf-8')
        except Exception as e:
            _log_error(name, 'read_transcript', e)
            failed += 1
            continue

        # Gemini SOAP extraction with retry
        result = None
        for attempt in range(1, 4):
            try:
                result = diarize_and_extract_soap_from_text(raw_text)
                break
            except Exception as e:
                err_str = str(e)
                if '503' in err_str and attempt < 3:
                    wait = 15 * attempt
                    print(f'  ⚠️  Gemini 503, retrying in {wait}s (attempt {attempt}/3)...')
                    time.sleep(wait)
                else:
                    _log_error(name, 'gemini_soap', e)
                    break

        if result is None:
            failed += 1
            continue

        row_df = _soap_to_row(result, name)
        if row_df is not None:
            if force and COMBINED_GROUND_TRUTH_CSV.exists():
                existing = pd.read_csv(COMBINED_GROUND_TRUTH_CSV)
                existing = existing[existing['audio_file'] != name]
                existing.to_csv(COMBINED_GROUND_TRUTH_CSV, index=False)
            _append_to_csv(row_df, COMBINED_GROUND_TRUTH_CSV)
            indiv_path = GROUND_TRUTH_DIR / f'{txt_path.stem}_soap.csv'
            row_df.to_csv(indiv_path, index=False)
            print(f'  ✅ Saved → {indiv_path.name}')
            success += 1
        else:
            _log_error(name, 'save', ValueError('SOAP result was None'))
            failed += 1

        if i < len(transcript_files):
            time.sleep(GEMINI_DELAY)

    print(f'\n{"="*60}')
    print(f'  Ground truth batch done: {success} ✅  {skipped} skipped  {failed} ❌')
    print(f'  Combined CSV: {COMBINED_GROUND_TRUTH_CSV}')
    print(f'{"="*60}')
    return success, skipped, failed

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Batch SOAP analysis — predicted (STT) and ground truth (clean transcripts)'
    )
    parser.add_argument(
        '--mode', choices=['both', 'predicted', 'ground_truth'], default='both',
        help='Which batch to run (default: both)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Process only the first N files (useful for testing)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-process files that already exist in the combined CSV and overwrite them. '
             'By default, already-processed files are skipped automatically.'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Deprecated alias for default behavior (files are skipped by default). Has no effect.'
    )
    parser.add_argument(
        '--audio-dir', default=str(AUDIO_DIR),
        help='Override audio files directory'
    )
    parser.add_argument(
        '--transcript-dir', default=str(TRANSCRIPT_DIR),
        help='Override clean transcripts directory'
    )
    args = parser.parse_args()

    audio_dir      = Path(args.audio_dir)
    transcript_dir = Path(args.transcript_dir)

    # Collect files
    audio_files     = sorted(audio_dir.glob('*.mp3'))
    transcript_files = sorted(transcript_dir.glob('*.txt'))

    if args.limit:
        audio_files      = audio_files[:args.limit]
        transcript_files = transcript_files[:args.limit]

    batch_start = time.time()
    total_success = total_failed = 0

    if args.mode in ('both', 'predicted'):
        if not audio_files:
            print(f'[WARN] No .mp3 files found in {audio_dir}')
        else:
            s, _, f = run_predicted_batch(audio_files, args.force)
            total_success += s; total_failed += f

    if args.mode in ('both', 'ground_truth'):
        if not transcript_files:
            print(f'[WARN] No .txt files found in {transcript_dir}')
        else:
            s, _, f = run_ground_truth_batch(transcript_files, args.force)
            total_success += s; total_failed += f

    elapsed = time.time() - batch_start
    print(f'\n🏁 All batches complete in {elapsed/60:.1f} min')
    print(f'   Total success: {total_success}  |  Total failed: {total_failed}')
    if total_failed:
        print(f'   Error log: {ERROR_LOG}')


if __name__ == '__main__':
    main()
