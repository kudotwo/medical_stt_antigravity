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
    """Transcribe audio locally with Whisper. Caches segments to JSON to avoid re-transcribing."""
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

def transcribe_audio_gemini(audio_path):
    """Transcribe audio using Gemini's native audio understanding (free tier, no GPU needed)."""
    import mimetypes
    client = genai.Client(api_key=GEMINI_API_KEY)
    mime_type = mimetypes.guess_type(audio_path)[0] or 'audio/mpeg'
    print(f'\n[1/3] Uploading audio to Gemini ({GEMINI_MODEL})...')
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
            model=GEMINI_MODEL,
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
        # Clean up uploaded file from Gemini servers
        try: client.files.delete(name=uploaded.name)
        except Exception: pass
SYSTEM_PROMPT = (
    'You are a clinical documentation assistant. You receive a raw STT transcript '
    'from a doctor-patient consultation as a numbered list of timestamped segments. '
    'The conversation may be in English, Bahasa Indonesia, or a mix of both.\n\n'

    'Tasks:\n'
    '1. Identify Doctor vs Patient from conversational context, regardless of language.\n'
    '   In Indonesian: the doctor (dokter) asks clinical questions; the patient (pasien) describes symptoms.\n'
    '2. Label every segment as "Doctor" or "Patient".\n'
    '3. Extract a structured SOAP record using ONLY evidence that is explicitly present in the transcript.\n\n'

    '=== LANGUAGE RULE (STRICT) ===\n'
    'Detect the SINGLE dominant language of the dialogue FIRST, before writing anything.\n'
    '- If the dialogue is in English, every free-text field (summary, chief_complaint, symptoms, '
    'diagnosis, treatment_plan, etc.) MUST be written entirely in English. Do NOT translate any '
    'part of it into Bahasa Indonesia or any other language.\n'
    '- If the dialogue is in Bahasa Indonesia, every free-text field MUST be written entirely in '
    'Bahasa Indonesia. Do NOT leave any part of it in English.\n'
    '- Never mix languages within a single field or across fields.\n'
    '- For enum values, translate them directly (encounter_type: \'konsultasi_awal\', \'tindak_lanjut\', '
    '\'tinjauan_hasil\', \'lainnya\'; extraction_confidence: \'tinggi\', \'sedang\', \'rendah\') ONLY if the '
    'dialogue is in Indonesian. If English, use the English enum values shown in the schema.\n'
    '- Do NOT translate the JSON field names/keys -- those always stay in English exactly as shown.\n\n'

    '=== EVIDENCE RULE (STRICT -- READ CAREFULLY) ===\n'
    'This is the most important rule. Your job is transcription-to-structure, NOT diagnosis.\n'
    '- Only include a fact if it was explicitly said by the Doctor or Patient in the transcript.\n'
    '- Do NOT infer, guess, or "complete" the clinical picture using general medical knowledge.\n'
    '- If a field has no explicit supporting evidence, leave it as null (single-value fields) or '
    '[] (list fields). An empty field is the CORRECT output when the conversation does not cover '
    'that topic -- do not pad it to look complete.\n'
    '- Never write placeholder text like "not specified", "unknown", or "unclear" -- use null/[] instead.\n\n'
    '- assessment.diagnosis: Fill this ONLY if the Doctor explicitly states a diagnosis, working '
    'diagnosis, or clinical impression out loud (e.g. "this sounds like...", "I think you have...", '
    '"my diagnosis is..."). Pure history-taking conversations often end with NO diagnosis spoken -- '
    'that is normal and expected. In that case return diagnosis: []. Restating the chief complaint '
    '(e.g. "chest pain, unspecified etiology") is NOT a diagnosis -- do not invent one.\n'
    '- assessment.differential_diagnosis: Unlike diagnosis, this field DOES call for standard clinical '
    'reasoning, the way a real clinician writes it on a note. List plausible differentials a clinician '
    'would consider given the symptoms, history, and risk factors ACTUALLY STATED in the transcript '
    '(e.g. stated family history, symptom character, described risk factors). Ground every item in '
    'something specific that was said. If the Doctor explicitly rules something out or lists '
    'possibilities out loud, include those too.\n'
    '- symptoms[]: Include every symptom explicitly described by the patient (or doctor summarizing '
    'the patient) -- primary AND secondary. This is about not missing what WAS said, never about '
    'adding what was not.\n'
    '- additional_notes: Social/environmental factors explicitly mentioned (smoking, living situation, '
    'work/school exposures, travel). null if none discussed.\n'
    '- plan.follow_up: The Doctor\'s exact closing instructions, ONLY if actually given. null otherwise.\n'
    '- objective.physical_exam_findings[]: Observable signs described, and exam findings the doctor '
    'explicitly states. Do not infer findings that would be "expected" for a diagnosis but were never '
    'actually mentioned.\n\n'

    '=== DRUG NAME NORMALIZATION ===\n'
    'The transcript comes from speech-to-text and drug names are often phonetically garbled '
    '(e.g. "lacetipro" means Lisinopril, "metaforin" means Metformin). Normalize a garbled name to its '
    'correct spelling ONLY when a medication was clearly mentioned. If uncertain, write the corrected '
    'name followed by "[STT-corrected]". Never add a medication that was not mentioned at all.\n\n'

    '=== SELF-CHECK BEFORE RESPONDING ===\n'
    'Before outputting JSON, silently re-check each non-empty field against the transcript: "Can I '
    'point to the exact line that supports this?" If not, set it back to null/[]. Confirm every '
    'free-text field is in the single dominant language, and that diagnosis was not invented in a '
    'conversation that never stated one.\n\n'
    
    '=== OTHERS ===\n'
    "If a segment's text appears to end one speaker's turn and begin another's, split it into two diarized_segments entries and estimate a reasonable time split."

    'Return ONLY valid JSON (no markdown fences, no explanation). Schema:\n'
    '{\n'
    '  "diarized_segments": [{"id":0,"speaker":"Doctor","start":0.0,"end":5.0,"text":"..."}],\n'
    '  "summary": "2-3 sentence clinical summary using only facts stated in the transcript (in conversation language)",\n'
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

def diarize_and_extract_soap_from_text(raw_text):
    print('\n[Live] Sending raw text to Gemini for diarization + SOAP extraction...')
    client = genai.Client(api_key=GEMINI_API_KEY)
    user_msg = 'Diarize the following raw text transcript (split it into logical Doctor and Patient turns) and extract SOAP.\n\nRAW TRANSCRIPT:\n' + raw_text
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

def save_results(result, audio_path, output_csv, output_transcript=None):
    audio_name = os.path.basename(audio_path)
    segs = result.get('diarized_segments', [])
    lines = [f'[{s.get("start",0):.1f}s->{s.get("end",0):.1f}s] {s.get("speaker","?")}: {s.get("text","").strip()}' for s in segs]
    txt = '\n'.join(lines)
    tp = output_transcript if output_transcript else audio_path.rsplit('.', 1)[0] + '_transcript.txt'
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

def run_pipeline(audio_path, output_csv=None, output_transcript=None, gemini_stt=False):
    pipeline_start = time.time()

    # --- Auto-generate timestamped output path if not specified ---
    if output_csv is None:
        os.makedirs('medical_reports', exist_ok=True)
        ts = datetime.now().strftime('%d_%m_%Y_%H_%M')
        output_csv = os.path.join('medical_reports', f'medical_report_stt_{ts}.csv')
        print(f'[Config] Output CSV     : {output_csv}')

    print('=' * 60)
    print('  Medical STT Pipeline  (Whisper + Gemini)')
    print('=' * 60)

    # --- Stage 1: Transcription ---
    t0 = time.time()
    if gemini_stt:
        print('  [Mode] Transcription: Gemini audio API (free tier)')
        segs = transcribe_audio_gemini(audio_path)
    else:
        print(f'  [Mode] Transcription: Local Whisper ({WHISPER_DEVICE})')
        segs = transcribe_audio(audio_path)
    if not segs:
        print('[ERROR] No segments from transcription.'); return
    print(f'  [Timer] Transcription : {time.time() - t0:.1f}s')

    # --- Stage 2: Diarization + SOAP ---
    t0 = time.time()
    result = diarize_and_extract_soap(segs)
    if result is None:
        print('[ERROR] Gemini extraction failed.'); return
    print(f'  [Timer] Gemini SOAP   : {time.time() - t0:.1f}s')

    # --- Stage 3: Save ---
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
    parser = argparse.ArgumentParser(description='Medical STT Pipeline (Whisper + Gemini)')
    parser.add_argument('audio_file', help='Path to the input audio file')
    parser.add_argument('--csv', default=None,
                        help='Output CSV path (default: medical_reports/medical_report_stt_DD_MM_YYYY_HH_MM.csv)')
    parser.add_argument('--transcript', default=None,
                        help='Output transcript .txt filename (default: <audio_file>_transcript.txt)')
    parser.add_argument('--gemini-stt', action='store_true',
                        help='Use Gemini audio API for transcription instead of local Whisper (faster, free tier available)')
    args = parser.parse_args()
    if not os.path.exists(args.audio_file):
        print(f'File not found: {args.audio_file}'); sys.exit(1)
    run_pipeline(args.audio_file, output_csv=args.csv, output_transcript=args.transcript, gemini_stt=args.gemini_stt)
