"""
Quick script to re-run Gemini diarization for specific cases
and save the diarized transcript files.
"""
import json, os, sys, time
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SOURCE_DIR))

# Set up CUDA libs before importing pipeline
_CUDA_LIB = SOURCE_DIR.parent / 'medical-stt-env/lib/python3.14/site-packages/nvidia/cu13/lib'
if _CUDA_LIB.exists():
    os.environ['LD_LIBRARY_PATH'] = str(_CUDA_LIB) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

from stt_pipeline_gpu import diarize_and_extract_soap

AUDIO_DIR = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Audio_Recordings'

# Cases that need diarized transcripts (CAR0001 already has one)
CASES = ['MSK0005', 'RES0050', 'GAS0003', 'DER0001']

for case_id in CASES:
    seg_path = AUDIO_DIR / f'{case_id}_segments.json'
    out_path = AUDIO_DIR / f'{case_id}_transcript.txt'

    if out_path.exists():
        print(f'[SKIP] {case_id} — transcript already exists')
        continue

    print(f'\n{"="*50}')
    print(f'Processing {case_id}...')

    with open(seg_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    result = diarize_and_extract_soap(segments)
    if result is None:
        print(f'  ❌ Failed for {case_id}')
        continue

    # Extract diarized segments and save transcript
    segs = result.get('diarized_segments', [])
    lines = [
        f'[{s.get("start", 0):.1f}s->{s.get("end", 0):.1f}s] {s.get("speaker", "?")}: {s.get("text", "").strip()}'
        for s in segs
    ]
    txt = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(txt)

    print(f'  ✅ Saved {len(segs)} diarized segments → {out_path.name}')
    print(f'  Preview: {txt[:200]}')

    # Rate limit
    time.sleep(3)

print('\n🏁 Done!')
