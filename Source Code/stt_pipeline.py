import json, os, shutil, sys, traceback

try:
    import imageio_ffmpeg
    _ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    _bin_dir    = os.path.dirname(_ffmpeg_bin)
    _ffmpeg_exe = os.path.join(_bin_dir, 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')
    if not os.path.exists(_ffmpeg_exe):
        shutil.copy(_ffmpeg_bin, _ffmpeg_exe)
    os.environ['PATH'] = _bin_dir + os.pathsep + os.environ['PATH']
except ImportError:
    pass

import whisper, torch, pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY not found in .env')

WHISPER_MODEL  = 'medium'
WHISPER_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GEMINI_MODEL   = 'gemini-3.1-flash-lite'

print(f'[Config] Whisper device : {WHISPER_DEVICE}')
print(f'[Config] Whisper model  : {WHISPER_MODEL}')
print(f'[Config] Gemini model   : {GEMINI_MODEL}')

def transcribe_audio(audio_path, use_cache=True):
    """Transcribe audio. Caches segments to JSON to avoid re-transcribing."""
    cache_path = audio_path.rsplit('.', 1)[0] + '_segments.json'
    if use_cache and os.path.exists(cache_path):
        print(f'\n[1/3] Loading cached segments from: {cache_path}')
        with open(cache_path, encoding='utf-8') as f:
            segments = json.load(f)
        print(f'      Segments loaded: {len(segments)}')
        return segments
    print(f'\n[1/3] Loading Whisper ({WHISPER_MODEL}) on {WHISPER_DEVICE}...')
    model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
    print(f'[2/3] Transcribing: {audio_path}')
    result = model.transcribe(audio_path, verbose=False)
    segments = result.get('segments', [])
    print(f'      Language : {result.get("language")}  |  Segments: {len(segments)}')
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2)
    print(f'      Segments cached to: {cache_path}')
    return segments

SYSTEM_PROMPT = (
    'You are a clinical documentation assistant. You receive a raw STT transcript '
    'from a doctor-patient consultation as a numbered list of timestamped segments. '
    'The conversation may be in English, Bahasa Indonesia, or a mix of both.\n\n'
    'Tasks:\n'
    '1. Identify Doctor vs Patient from conversational context, regardless of language.\n'
    '   In Indonesian: the doctor (dokter) asks clinical questions; the patient (pasien) describes symptoms.\n'
    '2. Label every segment as "Doctor" or "Patient".\n'
    '3. Extract a structured SOAP record. ALWAYS write all SOAP field values in English, '
    'even if the conversation is in Indonesian.\n\n'
    'Return ONLY valid JSON (no markdown fences, no explanation). Schema:\n'
    '{\n'
    '  "diarized_segments": [{"id":0,"speaker":"Doctor","start":0.0,"end":5.0,"text":"..."}],\n'
    '  "summary": "2-3 sentence clinical summary in English",\n'
    '  "encounter_type": "initial_consultation|follow_up|results_review|other",\n'
    '  "additional_notes": null,\n'
    '  "extraction_confidence": "high|medium|low",\n'
    '  "subjective": {"chief_complaint":"","symptoms":[],"symptom_onset":"","medical_history":[],"current_medications":[],"allergies":[]},\n'
    '  "objective": {"physical_exam_findings":[],"vital_signs":null},\n'
    '  "assessment": {"diagnosis":[],"differential_diagnosis":[]},\n'
    '  "plan": {"prescribed_medications":[],"treatment_plan":[],"investigations_ordered":[],"referrals":[],"follow_up":null}\n'
    '}'
)

def _fmt(segments):
    return '\n'.join(f'[{s["id"]}] ({s["start"]:.1f}s->{s["end"]:.1f}s) {s["text"].strip()}' for s in segments)

def diarize_and_extract_soap(segments):
    print('\n[3/3] Sending to Gemini for diarization + SOAP extraction...')
    client = genai.Client(api_key=GEMINI_API_KEY)
    user_msg = 'Label each segment as Doctor or Patient and extract SOAP.\n\nTRANSCRIPT:\n' + _fmt(segments)
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {'role':'user','parts':[{'text':SYSTEM_PROMPT}]},
                {'role':'model','parts':[{'text':'Understood. Producing SOAP JSON.'}]},
                {'role':'user','parts':[{'text':user_msg}]},
            ]
        )
        raw = resp.text.strip()
        if raw.startswith('`'):
            raw = raw.split('`')[1]
            if raw.startswith('json'): raw = raw[4:]
        raw = raw.strip()
        print(f'  RAW (first 200 chars): {raw[:200]}')
        return json.loads(raw)
    except Exception as e:
        print(f'[ERROR] {e}')
        traceback.print_exc()
        return None

def _flatten(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: ' | '.join(str(i) for i in x) if isinstance(x, list) else x)
    return df

def save_results(result, audio_path, output_csv='medical_report_stt.csv'):
    audio_name = os.path.basename(audio_path)
    segs = result.get('diarized_segments', [])
    lines = [f'[{s.get("start",0):.1f}s->{s.get("end",0):.1f}s] {s.get("speaker","?")}: {s.get("text","").strip()}' for s in segs]
    txt = '\n'.join(lines)
    tp = audio_path.rsplit('.', 1)[0] + '_transcript.txt'
    with open(tp, 'w', encoding='utf-8') as f: f.write(txt)
    print(f'\n  Transcript saved: {tp}')
    print('\n--- Diarized Transcript (preview) ---')
    print(txt[:800] + ('...' if len(txt) > 800 else ''))
    soap = {k:v for k,v in result.items() if k != 'diarized_segments'}
    flat = pd.json_normalize(soap, sep='.')
    flat = _flatten(flat)
    flat.insert(0, 'audio_file', audio_name)
    flat.to_csv(output_csv, index=False)
    print(f'\n  SOAP report saved: {output_csv}')
    print('\n--- SOAP (transposed) ---')
    print(flat.T.to_string())

def run_pipeline(audio_path, output_csv='medical_report_stt.csv'):
    print('=' * 60)
    print('  Medical STT Pipeline  (Whisper + Gemini)')
    print('=' * 60)
    segs = transcribe_audio(audio_path)
    if not segs:
        print('[ERROR] No segments from Whisper.'); return
    result = diarize_and_extract_soap(segs)
    if result is None:
        print('[ERROR] Gemini extraction failed.'); return
    save_results(result, audio_path, output_csv)
    print('\n' + '=' * 60)
    print('  Pipeline complete!')
    print('=' * 60)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python stt_pipeline.py <audio_file>'); sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f'File not found: {sys.argv[1]}'); sys.exit(1)
    run_pipeline(sys.argv[1])
