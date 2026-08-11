"""
stt_pipeline_gpu.py — GPU-accelerated variant of stt_pipeline.py
=================================================================
Replaces openai-whisper (fp32, OOMs on 4GB VRAM) with faster-whisper
(CTranslate2 float16), which uses ~1.4 GB VRAM vs ~3.4 GB.

Differences from stt_pipeline.py:
  - Uses `faster_whisper.WhisperModel` instead of `whisper.load_model`
  - compute_type = 'float16' -> half the VRAM, 2-4x faster throughput
  - Segment format is identical -> all GPT / SOAP logic is unchanged
  - All public function signatures are identical -> batch_soap_analysis.py
    can import from this file without modification

SOAP Extraction Model: OpenAI GPT-5 (gpt-5.6-terra)
  - Switched from Google Gemini (gemini-3.1-flash-lite) per professor recommendation
  - Same SOAP JSON schema and system prompt — only the API client changed

Hardware tested:
  - NVIDIA RTX 3050 Laptop GPU (3.68 GB usable VRAM)
  - faster-whisper medium float16 -> 1.42 GB VRAM used, 2.03 GB free
"""

import json, os, shutil, sys, time, traceback
from datetime import datetime

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

import ctypes
# Preload CUDA libraries so CTranslate2 finds libcublas.so.12 automatically.
# Linux only — Windows resolves CUDA DLLs automatically via PATH.
if sys.platform != 'win32':
    _cuda_lib_dir = os.path.join(os.path.dirname(sys.executable), '..', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'nvidia', 'cu13', 'lib')
    if os.path.exists(_cuda_lib_dir):
        for _lib in ['libcublasLt.so.12', 'libcublas.so.12', 'libcudart.so.12', 'libnvrtc.so.12']:
            _lib_path = os.path.join(_cuda_lib_dir, _lib)
            if os.path.exists(_lib_path):
                try:
                    ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

import torch, pandas as pd
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from openai import OpenAI
from google import genai  # kept for transcribe_audio_gemini (deprecated, Whisper is primary)

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError('OPENAI_API_KEY not found in .env — please add it and get the key from your professor')

# -- GPU / model config -------------------------------------------------------
WHISPER_MODEL   = 'medium'
WHISPER_DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
# float16 on GPU: ~1.4 GB VRAM (vs ~3.4 GB fp32 which OOMs on a 4 GB card).
# Falls back to int8 on CPU so the script still works without a GPU.
COMPUTE_TYPE    = 'float16' if WHISPER_DEVICE == 'cuda' else 'int8'
GPT_MODEL       = 'gpt-5.6-terra'  # switched from gemini-3.1-flash-lite per professor recommendation

print(f'[Config] Whisper device  : {WHISPER_DEVICE}')
print(f'[Config] Whisper model   : {WHISPER_MODEL}')
print(f'[Config] Compute type    : {COMPUTE_TYPE}')
print(f'[Config] GPT model       : {GPT_MODEL}')

if WHISPER_DEVICE == 'cuda':
    free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'[Config] VRAM free       : {free_gb:.2f} GB / {total_gb:.2f} GB total')

# Initialise the OpenAI client once (reused across all SOAP extraction calls)
_openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -- Lazy model loader (load once, reuse across batch) ------------------------
_whisper_model = None

def _get_whisper_model():
    """Load faster-whisper model once and cache it in memory."""
    global _whisper_model
    if _whisper_model is None:
        print(f'[Whisper] Loading {WHISPER_MODEL} on {WHISPER_DEVICE} ({COMPUTE_TYPE})...')
        t0 = time.time()
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        if WHISPER_DEVICE == 'cuda':
            used_gb = torch.cuda.memory_allocated(0) / 1024**3
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f'[Whisper] Loaded in {time.time()-t0:.1f}s | VRAM used: {used_gb:.2f} GB | free: {free_gb:.2f} GB')
        else:
            print(f'[Whisper] Loaded in {time.time()-t0:.1f}s (CPU mode)')
    return _whisper_model


# -- Transcription ------------------------------------------------------------

def transcribe_audio(audio_path, use_cache=True):
    """
    Transcribe audio with faster-whisper on GPU (float16).
    Returns a list of segment dicts compatible with the original stt_pipeline.py format:
      [{"id": 0, "start": 0.0, "end": 4.2, "text": "..."}]
    Caches result to JSON to avoid re-transcribing.
    """
    cache_path = audio_path.rsplit('.', 1)[0] + '_segments.json'
    if use_cache and os.path.exists(cache_path):
        print(f'\n[1/3] Loading cached segments from: {cache_path}')
        with open(cache_path, encoding='utf-8') as f:
            segments = json.load(f)
        print(f'      Segments loaded: {len(segments)}')
        return segments

    print(f'\n[1/3] Transcribing with faster-whisper ({WHISPER_MODEL}, {COMPUTE_TYPE})...')
    model = _get_whisper_model()

    # Medical vocabulary warm-up: biases the model's decoder toward clinical and
    # pharmaceutical terminology, significantly improving drug name recognition.
    MEDICAL_INITIAL_PROMPT = (
        'Doctor patient consultation. Medications: Lisinopril, Metformin, Rosuvastatin, '
        'Atorvastatin, Amlodipine, Omeprazole, Metoprolol, Amoxicillin, Azithromycin, '
        'Sertraline, Gabapentin, Hydrochlorothiazide, Losartan, Prednisone, Albuterol, '
        'Levothyroxine, Furosemide, Pantoprazole, Clopidogrel, Warfarin. '
        'Symptoms: hypertension, diabetes, dyspnea, tachycardia, bradycardia, '
        'myocardial infarction, pericarditis, pneumothorax. '
        'Dosage: milligrams, micrograms, twice daily, once daily, as needed.'
    )

    t0 = time.time()
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        initial_prompt=MEDICAL_INITIAL_PROMPT,  # biases decoder toward medical vocab
        vad_filter=True,          # skip silence -> faster + cleaner output
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Materialise the generator into a list (identical schema to openai-whisper)
    segments = [
        {
            'id':    i,
            'start': seg.start,
            'end':   seg.end,
            'text':  seg.text.strip(),
        }
        for i, seg in enumerate(segments_gen)
    ]

    elapsed = time.time() - t0
    print(f'      Language : {info.language} (prob={info.language_probability:.2f})')
    print(f'      Segments : {len(segments)} | Time: {elapsed:.1f}s')

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2)
    print(f'      Segments cached to: {cache_path}')
    return segments


def transcribe_audio_gemini(audio_path):
    """
    [DEPRECATED — Whisper GPU is the primary transcription path]
    Transcribe audio using Gemini's native audio understanding (free tier, no GPU needed).
    Kept for backwards compatibility but not used by the main pipeline or batch processor.
    """
    import mimetypes
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        print('[ERROR] GEMINI_API_KEY not set — transcribe_audio_gemini() is deprecated.')
        return []
    client = genai.Client(api_key=GEMINI_API_KEY)
    mime_type = mimetypes.guess_type(audio_path)[0] or 'audio/mpeg'
    print(f'\n[1/3] Uploading audio to Gemini (deprecated path)...')
    uploaded = client.files.upload(
        file=audio_path,
        config={'mime_type': mime_type}
    )
    print(f'      File uploaded: {uploaded.name}')
    prompt = (
        'Transcribe this doctor-patient audio recording. '
        'The conversation may be in English, Bahasa Indonesia, or mixed. '
        'Return ONLY a valid JSON array of segments with no markdown or explanation. '
        'Each segment: {"id": int, "start": float_seconds, "end": float_seconds, "text": "..."}. '
        'Estimate timestamps based on speech pacing. Example: '
        '[{"id":0,"start":0.0,"end":4.2,"text":"Selamat pagi, ada keluhan apa?"}]'
    )
    try:
        resp = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[uploaded, prompt]
        )
        raw = resp.text.strip()
        if raw.startswith('`'):
            raw = raw.split('`')[1]
            if raw.startswith('json'): raw = raw[4:]
        raw = raw.strip()
        segments = json.loads(raw)
        print(f'      Segments: {len(segments)}')
        return segments
    except Exception as e:
        print(f'[ERROR] Gemini STT failed: {e}')
        traceback.print_exc()
        return []
    finally:
        try: client.files.delete(name=uploaded.name)
        except Exception: pass


# -- Gemini SOAP extraction ---------------------------------------------------

SYSTEM_PROMPT = (
    'You are a clinical documentation assistant. You receive a raw STT transcript '
    'from a doctor-patient consultation as a numbered list of timestamped segments. '
    'The conversation may be in English, Bahasa Indonesia, or a mix of both.\n\n'
    'Tasks:\n'
    '1. Identify Doctor vs Patient from conversational context, regardless of language.\n'
    '   In Indonesian: the doctor (dokter) asks clinical questions; the patient (pasien) describes symptoms.\n'
    '2. Label every segment as "Doctor" or "Patient".\n'
    '3. Extract a structured SOAP record.\n'
    '   CRITICAL LANGUAGE RULE: The ENTIRE report (all values, including the summary, subjective, objective, '
    'assessment, and plan) MUST be written entirely in the SINGLE dominant language of the dialogue. '
    'Do NOT mix languages. If the dialogue is in Bahasa Indonesia, EVERY single field value (especially '
    'the summary) MUST be in Bahasa Indonesia. For enum values, translate them directly (e.g. for encounter_type use '
    '\'konsultasi_awal\', \'tindak_lanjut\', \'tinjauan_hasil\', \'lainnya\' and for extraction_confidence use \'tinggi\', \'sedang\', \'rendah\'). '
    'If English, use English. Do NOT translate the JSON field names/keys -- '
    'those must always stay in English exactly as shown in the schema.\n\n'
    'Extraction rules (follow strictly):\n'
    '- Be EXHAUSTIVE. Do not omit any clinical detail, even if mentioned only briefly or in passing.\n'
    '- symptoms[]: Include ALL symptoms the patient describes — primary AND secondary (e.g. fatigue, '
    'irritability, facial pressure, warmth/flushing). Do not stop at the chief complaint.\n'
    '- additional_notes: Use this field for social/environmental factors that may affect health: '
    'smoking history (who smokes, how much, indoor/outdoor), living situation, school/work exposures, '
    'household contacts, travel history, or any clinically relevant context not captured in SOAP fields. '
    'Set to null ONLY if none of these are present in the conversation.\n'
    '- differential_diagnosis[]: Include ALL alternative diagnoses the doctor explicitly mentions or '
    'reasons about, not just the final one. If the doctor rules something out (e.g. bacterial infection), '
    'still include it in the differential.\n'
    '- plan.follow_up: Capture the doctor\'s exact closing instructions — warning signs to watch for, '
    'when to return, and any conditional next steps (e.g. "return if symptoms worsen", '
    '"come back if she develops shortness of breath"). Do not leave this null if the doctor gives '
    'any closing guidance.\n'
    '- objective.physical_exam_findings[]: Include observable signs the patient or parent describes '
    '(e.g. facial flushing, warm to touch) even before a formal exam is done, as well as any '
    'exam procedures the doctor mentions performing.\n'
    '- current_medications[] and prescribed_medications[]: The transcript comes from speech-to-text '
    'and drug names are often phonetically garbled (e.g. "lacetipro" means Lisinopril, "metaforin" '
    'means Metformin, "rosuvistatin" means Rosuvastatin). Use your clinical knowledge to normalize '
    'any garbled drug name to its correct pharmaceutical spelling. If you are uncertain, write the '
    'corrected name followed by "[STT-corrected]" in brackets so it can be reviewed.\n\n'
    'Return ONLY valid JSON (no markdown fences, no explanation). Schema:\n'
    '{\n'
    '  "diarized_segments": [{"id":0,"speaker":"Doctor","start":0.0,"end":5.0,"text":"..."}],\n'
    '  "summary": "2-3 sentence clinical summary (in conversation language)",\n'
    '  "encounter_type": "initial_consultation|follow_up|results_review|other",\n'
    '  "additional_notes": "social/environmental context, or null if none",\n'
    '  "extraction_confidence": "high|medium|low",\n'
    '  "subjective": {"chief_complaint":"","symptoms":[],"symptom_onset":"","medical_history":[],"current_medications":[],"allergies":[]},\n'
    '  "objective": {"physical_exam_findings":[],"vital_signs":null},\n'
    '  "assessment": {"diagnosis":[],"differential_diagnosis":[]},\n'
    '  "plan": {"prescribed_medications":[],"treatment_plan":[],"investigations_ordered":[],"referrals":[],"follow_up":"closing instructions or null"}\n'
    '}'
)


def _fmt(segments):
    return '\n'.join(
        f'[{s["id"]}] ({s["start"]:.1f}s->{s["end"]:.1f}s) {s["text"].strip()}'
        for s in segments
    )


def diarize_and_extract_soap(segments):
    """Send STT segments to GPT for speaker diarization + SOAP extraction."""
    print(f'\n[3/3] Sending to GPT ({GPT_MODEL}) for diarization + SOAP extraction...')
    user_msg = 'Label each segment as Doctor or Patient and extract SOAP.\n\nTRANSCRIPT:\n' + _fmt(segments)
    try:
        response = _openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {'role': 'system',    'content': SYSTEM_PROMPT},
                {'role': 'assistant', 'content': 'Understood. Producing SOAP JSON.'},
                {'role': 'user',      'content': user_msg},
            ],
            response_format={'type': 'json_object'},
        )
        raw = response.choices[0].message.content.strip()
        print(f'  RAW (first 200 chars): {raw[:200]}')
        return json.loads(raw)
    except Exception as e:
        print(f'[ERROR] GPT SOAP extraction failed: {e}')
        traceback.print_exc()
        return None


def diarize_and_extract_soap_from_text(raw_text):
    """Send a plain-text transcript to GPT for diarization + SOAP extraction."""
    print(f'\n[Live] Sending raw text to GPT ({GPT_MODEL}) for diarization + SOAP extraction...')
    user_msg = 'Diarize the following raw text transcript (split it into logical Doctor and Patient turns) and extract SOAP.\n\nRAW TRANSCRIPT:\n' + raw_text
    try:
        response = _openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {'role': 'system',    'content': SYSTEM_PROMPT},
                {'role': 'assistant', 'content': 'Understood. Producing SOAP JSON.'},
                {'role': 'user',      'content': user_msg},
            ],
            response_format={'type': 'json_object'},
        )
        raw = response.choices[0].message.content.strip()
        print(f'  RAW (first 200 chars): {raw[:200]}')
        return json.loads(raw)
    except Exception as e:
        print(f'[ERROR] GPT SOAP extraction failed: {e}')
        traceback.print_exc()
        return None


# -- Result saving ------------------------------------------------------------

def _flatten(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: ' | '.join(str(i) for i in x) if isinstance(x, list) else x)
    return df


def save_results(result, audio_path, output_csv, output_transcript=None):
    audio_name = os.path.basename(audio_path)
    segs = result.get('diarized_segments', [])
    lines = [
        f'[{s.get("start", 0):.1f}s->{s.get("end", 0):.1f}s] {s.get("speaker", "?")}: {s.get("text", "").strip()}'
        for s in segs
    ]
    txt = '\n'.join(lines)
    tp = output_transcript if output_transcript else audio_path.rsplit('.', 1)[0] + '_transcript.txt'
    with open(tp, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'\n  Transcript saved: {tp}')
    print('\n--- Diarized Transcript (preview) ---')
    print(txt[:800] + ('...' if len(txt) > 800 else ''))

    soap = {k: v for k, v in result.items() if k != 'diarized_segments'}
    flat = pd.json_normalize(soap, sep='.')
    flat = _flatten(flat)
    flat.insert(0, 'audio_file', audio_name)
    flat.to_csv(output_csv, index=False)
    print(f'\n  SOAP report saved: {output_csv}')
    print('\n--- SOAP (transposed) ---')
    print(flat.T.to_string())


# -- Main pipeline entry point ------------------------------------------------

def run_pipeline(audio_path, output_csv=None, output_transcript=None, gemini_stt=False):
    pipeline_start = time.time()

    if output_csv is None:
        os.makedirs('medical_reports', exist_ok=True)
        ts = datetime.now().strftime('%d_%m_%Y_%H_%M')
        output_csv = os.path.join('medical_reports', f'medical_report_stt_{ts}.csv')
        print(f'[Config] Output CSV      : {output_csv}')

    print('=' * 60)
    print(f'  Medical STT Pipeline  (faster-whisper GPU + {GPT_MODEL})')
    print('=' * 60)

    # Stage 1 -- Transcription
    t0 = time.time()
    if gemini_stt:
        print('  [Mode] Transcription: Gemini audio API (free tier)')
        segs = transcribe_audio_gemini(audio_path)
    else:
        print(f'  [Mode] Transcription: faster-whisper ({WHISPER_DEVICE}, {COMPUTE_TYPE})')
        segs = transcribe_audio(audio_path)
    if not segs:
        print('[ERROR] No segments from transcription.'); return
    print(f'  [Timer] Transcription : {time.time() - t0:.1f}s')

    # Stage 2 -- Diarization + SOAP
    t0 = time.time()
    result = diarize_and_extract_soap(segs)
    if result is None:
        print('[ERROR] GPT extraction failed.'); return
    print(f'  [Timer] GPT SOAP      : {time.time() - t0:.1f}s')

    # Stage 3 -- Save
    t0 = time.time()
    save_results(result, audio_path, output_csv, output_transcript)
    print(f'  [Timer] Save results  : {time.time() - t0:.1f}s')

    total = time.time() - pipeline_start
    print('\n' + '=' * 60)
    print('  Pipeline complete!')
    print(f'  Total time : {total:.1f}s  ({total / 60:.1f} min)')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Medical STT Pipeline -- GPU (faster-whisper float16 + Gemini)')
    parser.add_argument('audio_file', help='Path to the input audio file')
    parser.add_argument('--csv', default=None,
                        help='Output CSV path (default: medical_reports/medical_report_stt_DD_MM_YYYY_HH_MM.csv)')
    parser.add_argument('--transcript', default=None,
                        help='Output transcript .txt filename (default: <audio_file>_transcript.txt)')
    parser.add_argument('--gemini-stt', action='store_true',
                        help='Use Gemini audio API for transcription instead of local faster-whisper')
    args = parser.parse_args()
    if not os.path.exists(args.audio_file):
        print(f'File not found: {args.audio_file}'); sys.exit(1)
    run_pipeline(args.audio_file, output_csv=args.csv, output_transcript=args.transcript, gemini_stt=args.gemini_stt)
