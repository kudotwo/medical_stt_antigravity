# """
# stt_pipeline_phi.py — Local GPU STT + Phi-4-mini SOAP pipeline
# ================================================================

# Based on stt_pipeline_gpu.py.

# Pipeline:
#     Audio
#       ↓
#     faster-whisper medium / float16
#       ↓
#     Ollama local API
#       ↓
#     Phi-4-mini
#       ↓
#     Speaker diarization + SOAP extraction
#       ↓
#     CSV + transcript

# The existing stt_pipeline.py and stt_pipeline_gpu.py are NOT modified.

# Requirements:
#     - Ollama installed and running
#     - phi4-mini pulled in Ollama
#     - faster-whisper
#     - torch
#     - pandas
#     - python-dotenv
#     - imageio-ffmpeg

# Ollama endpoint:
#     http://127.0.0.1:11434

# Default model:
#     phi4-mini
# """

# import json
# import os
# import shutil
# import sys
# import time
# import traceback
# import gc
# import urllib.request
# import urllib.error
# from datetime import datetime


# # ============================================================================
# # FFmpeg setup
# # ============================================================================

# try:
#     import imageio_ffmpeg

#     _ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
#     _bin_dir = os.path.dirname(_ffmpeg_bin)
#     _ffmpeg_exe = os.path.join(
#         _bin_dir,
#         'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'
#     )

#     if not os.path.exists(_ffmpeg_exe):
#         shutil.copy(_ffmpeg_bin, _ffmpeg_exe)

#     os.environ['PATH'] = _bin_dir + os.pathsep + os.environ['PATH']

# except ImportError:
#     pass


# # ============================================================================
# # CUDA library setup
# # ============================================================================

# import ctypes

# # Linux only — preload CUDA libraries so CTranslate2 can find them.
# if sys.platform != 'win32':

#     _cuda_lib_dir = os.path.join(
#         os.path.dirname(sys.executable),
#         '..',
#         'lib',
#         f'python{sys.version_info.major}.{sys.version_info.minor}',
#         'site-packages',
#         'nvidia',
#         'cu13',
#         'lib'
#     )

#     if os.path.exists(_cuda_lib_dir):

#         for _lib in [
#             'libcublasLt.so.12',
#             'libcublas.so.12',
#             'libcudart.so.12',
#             'libnvrtc.so.12'
#         ]:

#             _lib_path = os.path.join(_cuda_lib_dir, _lib)

#             if os.path.exists(_lib_path):
#                 try:
#                     ctypes.CDLL(
#                         _lib_path,
#                         mode=ctypes.RTLD_GLOBAL
#                     )
#                 except Exception:
#                     pass


# # ============================================================================
# # Imports
# # ============================================================================

# import torch
# import pandas as pd

# from faster_whisper import WhisperModel
# from dotenv import load_dotenv


# # ============================================================================
# # Configuration
# # ============================================================================

# load_dotenv()

# WHISPER_MODEL = 'medium'

# WHISPER_DEVICE = (
#     'cuda'
#     if torch.cuda.is_available()
#     else 'cpu'
# )

# COMPUTE_TYPE = (
#     'float16'
#     if WHISPER_DEVICE == 'cuda'
#     else 'int8'
# )

# # Ollama configuration
# OLLAMA_URL = os.environ.get(
#     'OLLAMA_URL',
#     'http://127.0.0.1:11434'
# )

# OLLAMA_MODEL = os.environ.get(
#     'OLLAMA_MODEL',
#     'phi4-mini'
# )

# # Keep context reasonably sized for your 4 GB VRAM GPU.
# # We can increase this later if necessary.
# OLLAMA_NUM_CTX = int(
#     os.environ.get('OLLAMA_NUM_CTX', '8192')
# )

# # Temperature 0 is desirable for this extraction task.
# OLLAMA_TEMPERATURE = float(
#     os.environ.get('OLLAMA_TEMPERATURE', '0')
# )


# print(f'[Config] Whisper device : {WHISPER_DEVICE}')
# print(f'[Config] Whisper model  : {WHISPER_MODEL}')
# print(f'[Config] Compute type   : {COMPUTE_TYPE}')
# print(f'[Config] Ollama URL     : {OLLAMA_URL}')
# print(f'[Config] Ollama model   : {OLLAMA_MODEL}')
# print(f'[Config] Ollama context : {OLLAMA_NUM_CTX}')
# print(f'[Config] Temperature    : {OLLAMA_TEMPERATURE}')


# if WHISPER_DEVICE == 'cuda':

#     free_gb = torch.cuda.mem_get_info()[0] / 1024**3
#     total_gb = (
#         torch.cuda.get_device_properties(0).total_memory
#         / 1024**3
#     )

#     print(
#         f'[Config] VRAM free      : '
#         f'{free_gb:.2f} GB / {total_gb:.2f} GB total'
#     )


# # ============================================================================
# # Whisper lazy loader
# # ============================================================================

# _whisper_model = None


# def _get_whisper_model():
#     """
#     Load faster-whisper once and cache it.
#     """

#     global _whisper_model

#     if _whisper_model is None:

#         print(
#             f'[Whisper] Loading {WHISPER_MODEL} '
#             f'on {WHISPER_DEVICE} ({COMPUTE_TYPE})...'
#         )

#         t0 = time.time()

#         _whisper_model = WhisperModel(
#             WHISPER_MODEL,
#             device=WHISPER_DEVICE,
#             compute_type=COMPUTE_TYPE,
#         )

#         if WHISPER_DEVICE == 'cuda':

#             used_gb = (
#                 torch.cuda.memory_allocated(0)
#                 / 1024**3
#             )

#             free_gb = (
#                 torch.cuda.mem_get_info()[0]
#                 / 1024**3
#             )

#             print(
#                 f'[Whisper] Loaded in '
#                 f'{time.time() - t0:.1f}s | '
#                 f'VRAM used: {used_gb:.2f} GB | '
#                 f'free: {free_gb:.2f} GB'
#             )

#         else:

#             print(
#                 f'[Whisper] Loaded in '
#                 f'{time.time() - t0:.1f}s '
#                 f'(CPU mode)'
#             )

#     return _whisper_model


# # ============================================================================
# # Optional Whisper GPU release
# # ============================================================================

# def _release_whisper_model():
#     """
#     Release faster-whisper before starting Phi-4.

#     Your RTX 3050 has 4 GB VRAM.

#     faster-whisper medium float16:
#         ~1.4 GB VRAM

#     Phi-4-mini:
#         ~2.7 GB VRAM

#     Keeping both loaded simultaneously can exceed available VRAM.

#     For this local pipeline, we therefore release Whisper after
#     transcription and before calling Phi-4.

#     This means Whisper will be reloaded if another transcription
#     is requested later in the same Python process.
#     """

#     global _whisper_model

#     if _whisper_model is None:
#         return

#     print('[Whisper] Releasing Whisper model from memory...')

#     try:
#         del _whisper_model
#     except Exception:
#         pass

#     _whisper_model = None

#     gc.collect()

#     if torch.cuda.is_available():

#         try:
#             torch.cuda.empty_cache()
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass

#     time.sleep(0.5)

#     if torch.cuda.is_available():

#         free_gb = (
#             torch.cuda.mem_get_info()[0]
#             / 1024**3
#         )

#         total_gb = (
#             torch.cuda.get_device_properties(0).total_memory
#             / 1024**3
#         )

#         print(
#             f'[Whisper] GPU memory after release: '
#             f'{free_gb:.2f} GB free / '
#             f'{total_gb:.2f} GB total'
#         )


# # ============================================================================
# # Transcription
# # ============================================================================

# def transcribe_audio(audio_path, use_cache=True):
#     """
#     Transcribe audio using faster-whisper.

#     Returns:
#         [
#             {
#                 "id": 0,
#                 "start": 0.0,
#                 "end": 4.2,
#                 "text": "..."
#             }
#         ]

#     Results are cached to:
#         <audio_name>_segments.json
#     """

#     cache_path = (
#         audio_path.rsplit('.', 1)[0]
#         + '_segments.json'
#     )

#     if use_cache and os.path.exists(cache_path):

#         print(
#             f'\n[1/3] Loading cached segments from: '
#             f'{cache_path}'
#         )

#         with open(
#             cache_path,
#             encoding='utf-8'
#         ) as f:

#             segments = json.load(f)

#         print(
#             f'      Segments loaded: '
#             f'{len(segments)}'
#         )

#         return segments


#     print(
#         f'\n[1/3] Transcribing with faster-whisper '
#         f'({WHISPER_MODEL}, {COMPUTE_TYPE})...'
#     )

#     model = _get_whisper_model()


#     # Medical vocabulary warm-up.
#     MEDICAL_INITIAL_PROMPT = (
#         'Doctor patient consultation. '
#         'Medications: Lisinopril, Metformin, Rosuvastatin, '
#         'Atorvastatin, Amlodipine, Omeprazole, Metoprolol, '
#         'Amoxicillin, Azithromycin, Sertraline, Gabapentin, '
#         'Hydrochlorothiazide, Losartan, Prednisone, Albuterol, '
#         'Levothyroxine, Furosemide, Pantoprazole, Clopidogrel, '
#         'Warfarin. '
#         'Symptoms: hypertension, diabetes, dyspnea, tachycardia, '
#         'bradycardia, myocardial infarction, pericarditis, '
#         'pneumothorax. '
#         'Dosage: milligrams, micrograms, twice daily, '
#         'once daily, as needed.'
#     )


#     t0 = time.time()

#     segments_gen, info = model.transcribe(
#         audio_path,
#         beam_size=5,
#         initial_prompt=MEDICAL_INITIAL_PROMPT,
#         vad_filter=True,
#         vad_parameters=dict(
#             min_silence_duration_ms=500
#         ),
#     )


#     segments = [
#         {
#             'id': i,
#             'start': seg.start,
#             'end': seg.end,
#             'text': seg.text.strip(),
#         }
#         for i, seg in enumerate(segments_gen)
#     ]


#     elapsed = time.time() - t0

#     print(
#         f'      Language : '
#         f'{info.language} '
#         f'(prob={info.language_probability:.2f})'
#     )

#     print(
#         f'      Segments : '
#         f'{len(segments)} | '
#         f'Time: {elapsed:.1f}s'
#     )


#     with open(
#         cache_path,
#         'w',
#         encoding='utf-8'
#     ) as f:

#         json.dump(
#             segments,
#             f,
#             indent=2
#         )


#     print(
#         f'      Segments cached to: '
#         f'{cache_path}'
#     )

#     return segments


# # ============================================================================
# # SOAP prompt
# # ============================================================================
# #
# # This is intentionally kept equivalent to the prompt in
# # stt_pipeline_gpu.py.
# #
# # We want the model comparison to be:
# #
# #     GPT-5.6 vs Phi-4-mini
# #
# # rather than:
# #
# #     different model + different prompt
# #
# # ============================================================================

# SYSTEM_PROMPT = (
#     'You are a clinical documentation assistant. You receive a raw STT transcript '
#     'from a doctor-patient consultation as a numbered list of timestamped segments. '
#     'The conversation may be in English, Bahasa Indonesia, or a mix of both.\n\n'

#     'Tasks:\n'
#     '1. Identify Doctor vs Patient from conversational context, regardless of language.\n'
#     '   In Indonesian: the doctor (dokter) asks clinical questions; the patient (pasien) describes symptoms.\n'
#     '2. Label every segment as "Doctor" or "Patient".\n'
#     '3. Extract a structured SOAP record using ONLY evidence that is explicitly present in the transcript.\n\n'

#     '=== LANGUAGE RULE (STRICT) ===\n'
#     'Detect the SINGLE dominant language of the dialogue FIRST, before writing anything.\n'
#     '- If the dialogue is in English, every free-text field (summary, chief_complaint, symptoms, '
#     'diagnosis, treatment_plan, etc.) MUST be written entirely in English. Do NOT translate any '
#     'part of it into Bahasa Indonesia or any other language.\n'
#     '- If the dialogue is in Bahasa Indonesia, every free-text field MUST be written entirely in '
#     'Bahasa Indonesia. Do NOT leave any part of it in English.\n'
#     '- Never mix languages within a single field or across fields.\n'
#     '- For enum values, translate them directly (encounter_type: \'konsultasi_awal\', \'tindak_lanjut\', '
#     '\'tinjauan_hasil\', \'lainnya\'; extraction_confidence: \'tinggi\', \'sedang\', \'rendah\') ONLY if the '
#     'dialogue is in Indonesian. If English, use the English enum values shown in the schema.\n'
#     '- Do NOT translate the JSON field names/keys -- those always stay in English exactly as shown.\n\n'

#     '=== EVIDENCE RULE (STRICT -- READ CAREFULLY) ===\n'
#     'This is the most important rule. Your job is transcription-to-structure, NOT diagnosis.\n'
#     '- Only include a fact if it was explicitly said by the Doctor or Patient in the transcript.\n'
#     '- Do NOT infer, guess, or "complete" the clinical picture using general medical knowledge.\n'
#     '- If a field has no explicit supporting evidence, leave it as null (single-value fields) or '
#     '[] (list fields). An empty field is the CORRECT output when the conversation does not cover '
#     'that topic -- do not pad it to look complete.\n'
#     '- Never write placeholder text like "not specified", "unknown", or "unclear" -- use null/[] instead.\n\n'

#     '- assessment.diagnosis: Fill this ONLY if the Doctor explicitly states a diagnosis, working '
#     'diagnosis, or clinical impression out loud (e.g. "this sounds like...", "I think you have...", '
#     '"my diagnosis is..."). Pure history-taking conversations often end with NO diagnosis spoken -- '
#     'that is normal and expected. In that case return diagnosis: []. Restating the chief complaint '
#     '(e.g. "chest pain, unspecified etiology") is NOT a diagnosis -- do not invent one.\n'

#     '- assessment.differential_diagnosis: Unlike diagnosis, this field DOES call for standard clinical '
#     'reasoning, the way a real clinician writes it on a note. List plausible differentials a clinician '
#     'would consider given the symptoms, history, and risk factors ACTUALLY STATED in the transcript '
#     '(e.g. stated family history, symptom character, described risk factors). Ground every item in '
#     'something specific that was said. If the Doctor explicitly rules something out or lists '
#     'possibilities out loud, include those too.\n'

#     '- symptoms[]: Include every symptom explicitly described by the patient (or doctor summarizing '
#     'the patient) -- primary AND secondary. This is about not missing what WAS said, never about '
#     'adding what was not.\n'

#     '- additional_notes: Social/environmental factors explicitly mentioned (smoking, living situation, '
#     'work/school exposures, travel). null if none discussed.\n'

#     '- plan.follow_up: The Doctor\'s exact closing instructions, ONLY if actually given. null otherwise.\n'

#     '- objective.physical_exam_findings[]: Observable signs described, and exam findings the doctor '
#     'explicitly states. Do not infer findings that would be "expected" for a diagnosis but were never '
#     'actually mentioned.\n\n'

#     '=== DRUG NAME NORMALIZATION ===\n'
#     'The transcript comes from speech-to-text and drug names are often phonetically garbled '
#     '(e.g. "lacetipro" means Lisinopril, "metaforin" means Metformin). Normalize a garbled name to its '
#     'correct spelling ONLY when a medication was clearly mentioned. If uncertain, write the corrected '
#     'name followed by "[STT-corrected]". Never add a medication that was not mentioned at all.\n\n'

#     '=== SELF-CHECK BEFORE RESPONDING ===\n'
#     'Before outputting JSON, silently re-check each non-empty field against the transcript: "Can I '
#     'point to the exact line that supports this?" If not, set it back to null/[]. Confirm every '
#     'free-text field is in the single dominant language, and that diagnosis was not invented in a '
#     'conversation that never stated one.\n\n'

#     '=== OTHERS ===\n'
#     "If a segment's text appears to end one speaker's turn and begin another's, split it into two "
#     "diarized_segments entries and estimate a reasonable time split."

#     '\n'
#     'Return ONLY valid JSON (no markdown fences, no explanation). Schema:\n'

#     '{\n'
#     '  "diarized_segments": [{"id":0,"speaker":"Doctor","start":0.0,"end":5.0,"text":"..."}],\n'
#     '  "summary": "2-3 sentence clinical summary using only facts stated in the transcript (in conversation language)",\n'
#     '  "encounter_type": "initial_consultation|follow_up|results_review|other",\n'
#     '  "additional_notes": "social/environmental context, or null if none",\n'
#     '  "extraction_confidence": "high|medium|low",\n'
#     '  "subjective": {"chief_complaint":"","symptoms":[],"symptom_onset":"","medical_history":[],"current_medications":[],"allergies":[]},\n'
#     '  "objective": {"physical_exam_findings":[],"vital_signs":null},\n'
#     '  "assessment": {"diagnosis":[],"differential_diagnosis":[]},\n'
#     '  "plan": {"prescribed_medications":[],"treatment_plan":[],"investigations_ordered":[],"referrals":[],"follow_up":"closing instructions or null"}\n'
#     '}'
# )


# # ============================================================================
# # Formatting
# # ============================================================================

# def _fmt(segments):
#     """
#     Convert Whisper segments into the format expected by the SOAP prompt.
#     """

#     return '\n'.join(
#         f'[{s["id"]}] '
#         f'({s["start"]:.1f}s->{s["end"]:.1f}s) '
#         f'{s["text"].strip()}'
#         for s in segments
#     )


# # ============================================================================
# # Ollama API
# # ============================================================================

# def _ollama_chat(system_prompt, user_prompt):
#     """
#     Send a chat request to the locally running Ollama server.

#     Uses Python's built-in urllib so no additional Ollama Python package
#     is required.

#     Returns:
#         Parsed JSON dictionary.
#     """

#     url = OLLAMA_URL.rstrip('/') + '/api/chat'


#     payload = {
#         'model': OLLAMA_MODEL,

#         'messages': [
#             {
#                 'role': 'system',
#                 'content': system_prompt,
#             },
#             {
#                 'role': 'user',
#                 'content': user_prompt,
#             },
#         ],

#         # Ollama supports structured JSON output with format="json".
#         'format': 'json',

#         # Do not stream because we need the complete JSON response.
#         'stream': False,

#         'options': {
#             'temperature': OLLAMA_TEMPERATURE,
#             'num_ctx': OLLAMA_NUM_CTX,
#         },
#     }


#     body = json.dumps(payload).encode('utf-8')


#     request = urllib.request.Request(
#         url,
#         data=body,
#         headers={
#             'Content-Type': 'application/json',
#         },
#         method='POST',
#     )


#     t0 = time.time()


#     try:

#         with urllib.request.urlopen(
#             request,
#             timeout=600
#         ) as response:

#             response_body = response.read().decode(
#                 'utf-8'
#             )


#         elapsed = time.time() - t0

#         data = json.loads(response_body)

#         print(
#             f'[Ollama] Response received in '
#             f'{elapsed:.1f}s'
#         )


#         # Ollama /api/chat response:
#         #
#         # {
#         #   "message": {
#         #       "role": "assistant",
#         #       "content": "..."
#         #   }
#         # }

#         message = data.get('message', {})

#         raw = message.get(
#             'content',
#             ''
#         ).strip()


#         if not raw:
#             raise ValueError(
#                 'Ollama returned an empty response.'
#             )


#         print(
#             f'  RAW (first 500 chars): '
#             f'{raw[:500]}'
#         )


#         # Parse JSON returned by Phi.
#         result = json.loads(raw)

#         return result


#     except urllib.error.URLError as e:

#         raise RuntimeError(
#             f'Could not connect to Ollama at '
#             f'{url}. '
#             f'Is "ollama serve" running? '
#             f'Original error: {e}'
#         ) from e


# # ============================================================================
# # SOAP extraction
# # ============================================================================

# def diarize_and_extract_soap(segments):
#     """
#     Send timestamped Whisper segments to Phi-4-mini through Ollama.

#     This replaces the GPT-5.6 call from stt_pipeline_gpu.py.
#     """

#     print(
#         f'\n[3/3] Sending to Ollama '
#         f'({OLLAMA_MODEL}) for '
#         f'diarization + SOAP extraction...'
#     )


#     user_msg = (
#         'Label each segment as Doctor or Patient '
#         'and extract SOAP.\n\n'
#         'TRANSCRIPT:\n'
#         + _fmt(segments)
#     )


#     try:

#         result = _ollama_chat(
#             SYSTEM_PROMPT,
#             user_msg
#         )

#         return result


#     except Exception as e:

#         print(
#             f'[ERROR] Phi-4-mini SOAP extraction failed: '
#             f'{e}'
#         )

#         traceback.print_exc()

#         return None


# # ============================================================================
# # Raw text SOAP extraction
# # ============================================================================

# def diarize_and_extract_soap_from_text(raw_text):
#     """
#     Send a plain-text transcript to Phi-4-mini.

#     This function has the same name/signature as the function used by
#     the existing server.py.

#     NOTE:
#         This function is intended for LOCAL testing with Ollama.

#         It cannot be used by your current Vercel deployment unless
#         Ollama/Phi-4 is hosted somewhere reachable by the Vercel server.
#     """

#     print(
#         f'\n[Live] Sending raw text to Ollama '
#         f'({OLLAMA_MODEL}) for '
#         f'diarization + SOAP extraction...'
#     )


#     user_msg = (
#         'Diarize the following raw text transcript '
#         '(split it into logical Doctor and Patient turns) '
#         'and extract SOAP.\n\n'
#         'RAW TRANSCRIPT:\n'
#         + raw_text
#     )


#     try:

#         result = _ollama_chat(
#             SYSTEM_PROMPT,
#             user_msg
#         )

#         return result


#     except Exception as e:

#         print(
#             f'[ERROR] Phi-4-mini SOAP extraction failed: '
#             f'{e}'
#         )

#         traceback.print_exc()

#         return None


# # ============================================================================
# # DataFrame helpers
# # ============================================================================

# def _flatten(df):
#     """
#     Convert list-valued DataFrame cells into readable strings.
#     """

#     for col in df.columns:

#         if df[col].apply(
#             lambda x: isinstance(x, list)
#         ).any():

#             df[col] = df[col].apply(
#                 lambda x:
#                     ' | '.join(str(i) for i in x)
#                     if isinstance(x, list)
#                     else x
#             )

#     return df


# # ============================================================================
# # Save results
# # ============================================================================

# def save_results(
#     result,
#     audio_path,
#     output_csv,
#     output_transcript=None
# ):
#     """
#     Save diarized transcript and SOAP result.
#     """

#     audio_name = os.path.basename(
#         audio_path
#     )


#     segs = result.get(
#         'diarized_segments',
#         []
#     )


#     lines = [
#         (
#             f'[{s.get("start", 0):.1f}s->'
#             f'{s.get("end", 0):.1f}s] '
#             f'{s.get("speaker", "?")}: '
#             f'{s.get("text", "").strip()}'
#         )
#         for s in segs
#     ]


#     txt = '\n'.join(lines)


#     tp = (
#         output_transcript
#         if output_transcript
#         else audio_path.rsplit('.', 1)[0]
#         + '_transcript.txt'
#     )


#     with open(
#         tp,
#         'w',
#         encoding='utf-8'
#     ) as f:

#         f.write(txt)


#     print(
#         f'\n  Transcript saved: {tp}'
#     )


#     print(
#         '\n--- Diarized Transcript '
#         '(preview) ---'
#     )

#     print(
#         txt[:800]
#         + ('...' if len(txt) > 800 else '')
#     )


#     soap = {
#         k: v
#         for k, v in result.items()
#         if k != 'diarized_segments'
#     }


#     flat = pd.json_normalize(
#         soap,
#         sep='.'
#     )


#     flat = _flatten(flat)


#     flat.insert(
#         0,
#         'audio_file',
#         audio_name
#     )


#     flat.to_csv(
#         output_csv,
#         index=False
#     )


#     print(
#         f'\n  SOAP report saved: '
#         f'{output_csv}'
#     )


#     print(
#         '\n--- SOAP (transposed) ---'
#     )

#     print(
#         flat.T.to_string()
#     )


# # ============================================================================
# # Main pipeline
# # ============================================================================

# def run_pipeline(
#     audio_path,
#     output_csv=None,
#     output_transcript=None,
#     gemini_stt=False
# ):
#     """
#     Run:

#         Audio
#           ↓
#         faster-whisper
#           ↓
#         Phi-4-mini / Ollama
#           ↓
#         SOAP
#           ↓
#         CSV

#     gemini_stt is retained in the function signature for compatibility
#     with the other pipeline files, but is not used here.
#     """

#     pipeline_start = time.time()


#     if gemini_stt:

#         print(
#             '[Warning] --gemini-stt was supplied, '
#             'but this Phi pipeline does not use Gemini.'
#         )


#     if output_csv is None:

#         os.makedirs(
#             'medical_reports',
#             exist_ok=True
#         )


#         ts = datetime.now().strftime(
#             '%d_%m_%Y_%H_%M'
#         )


#         output_csv = os.path.join(
#             'medical_reports',
#             f'medical_report_stt_phi_'
#             f'{ts}.csv'
#         )


#         print(
#             f'[Config] Output CSV: '
#             f'{output_csv}'
#         )


#     print('=' * 60)

#     print(
#         '  Medical STT Pipeline'
#     )

#     print(
#         '  faster-whisper GPU + '
#         'Phi-4-mini / Ollama'
#     )

#     print('=' * 60)


#     # ------------------------------------------------------------------------
#     # Stage 1 — Transcription
#     # ------------------------------------------------------------------------

#     t0 = time.time()


#     print(
#         f'  [Mode] Transcription: '
#         f'faster-whisper '
#         f'({WHISPER_DEVICE}, {COMPUTE_TYPE})'
#     )


#     segs = transcribe_audio(
#         audio_path
#     )


#     if not segs:

#         print(
#             '[ERROR] No segments '
#             'from transcription.'
#         )

#         return


#     print(
#         f'  [Timer] Transcription: '
#         f'{time.time() - t0:.1f}s'
#     )


#     # ------------------------------------------------------------------------
#     # Stage 1.5 — Release Whisper GPU memory
#     # ------------------------------------------------------------------------

#     #
#     # This is important on your RTX 3050.
#     #
#     # Whisper medium float16 ≈ 1.4 GB VRAM
#     # Phi-4-mini ≈ 2.7 GB VRAM
#     #
#     # Releasing Whisper gives Phi-4 the best chance of remaining
#     # fully GPU accelerated.
#     #

#     _release_whisper_model()


#     # ------------------------------------------------------------------------
#     # Stage 2 — Phi-4-mini SOAP extraction
#     # ------------------------------------------------------------------------

#     t0 = time.time()


#     result = diarize_and_extract_soap(
#         segs
#     )


#     if result is None:

#         print(
#             '[ERROR] Phi-4-mini '
#             'SOAP extraction failed.'
#         )

#         return


#     print(
#         f'  [Timer] Phi-4 SOAP: '
#         f'{time.time() - t0:.1f}s'
#     )


#     # ------------------------------------------------------------------------
#     # Stage 3 — Save
#     # ------------------------------------------------------------------------

#     t0 = time.time()


#     save_results(
#         result,
#         audio_path,
#         output_csv,
#         output_transcript
#     )


#     print(
#         f'  [Timer] Save results: '
#         f'{time.time() - t0:.1f}s'
#     )


#     # ------------------------------------------------------------------------
#     # Complete
#     # ------------------------------------------------------------------------

#     total = (
#         time.time()
#         - pipeline_start
#     )


#     print(
#         '\n' + '=' * 60
#     )

#     print(
#         '  Pipeline complete!'
#     )

#     print(
#         f'  Total time: '
#         f'{total:.1f}s '
#         f'({total / 60:.1f} min)'
#     )

#     print(
#         '=' * 60
#     )


# # ============================================================================
# # Command-line interface
# # ============================================================================

# if __name__ == '__main__':

#     import argparse


#     parser = argparse.ArgumentParser(
#         description=(
#             'Medical STT Pipeline — '
#             'faster-whisper GPU + '
#             'Phi-4-mini via Ollama'
#         )
#     )


#     parser.add_argument(
#         'audio_file',
#         help='Path to the input audio file'
#     )


#     parser.add_argument(
#         '--csv',
#         default=None,
#         help=(
#             'Output CSV path '
#             '(default: medical_reports/'
#             'medical_report_stt_phi_'
#             'DD_MM_YYYY_HH_MM.csv)'
#         )
#     )


#     parser.add_argument(
#         '--transcript',
#         default=None,
#         help=(
#             'Output transcript .txt filename '
#             '(default: <audio_file>_transcript.txt)'
#         )
#     )


#     args = parser.parse_args()


#     if not os.path.exists(
#         args.audio_file
#     ):

#         print(
#             f'File not found: '
#             f'{args.audio_file}'
#         )

#         sys.exit(1)


#     run_pipeline(
#         args.audio_file,
#         output_csv=args.csv,
#         output_transcript=args.transcript
#     )


#Shorter version (no audio transcription)
import json
import urllib.request
import time


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "phi4-mini"


SYSTEM_PROMPT = """

    You are a medical documentation assistant.

Use ONLY information explicitly present in the transcript.

Return JSON:

{
  "chief_complaint":"",
  "symptoms":[],
  "diagnosis":[]
}

Do not infer information.
Do not add tests.
Do not add treatment.
Do not add medical history unless stated.
"""


def format_segments(segments):
    return "\n".join(
        f'[{s["id"]}] ({s["start"]:.1f}s->{s["end"]:.1f}s) '
        f'{s["text"].strip()}'
        for s in segments
    )


def ask_phi(transcript):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content":
                    "Label each segment as Doctor or Patient "
                    "and extract SOAP.\n\n"
                    "TRANSCRIPT:\n"
                    + transcript
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    start = time.time()

    with urllib.request.urlopen(
        request,
        timeout=600
    ) as response:

        data = json.loads(
            response.read().decode("utf-8")
        )

    print(
        f"Phi-4 response time: "
        f"{time.time() - start:.1f}s"
    )

    raw = data["message"]["content"]

    return json.loads(raw)


def main():

    with open(
        "audio_sample/audio_recordings/Audio_Recordings/CAR0001_segments.json",
        "r",
        encoding="utf-8"
    ) as f:

        segments = json.load(f)

    print(
        f"Loaded {len(segments)} segments."
    )

    transcript = format_segments(
        segments
    )

    result = ask_phi(
        transcript
    )

    with open(
        "car0001_phi_soap_promptv2.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nSaved:"
        "\ncar0001_phi_soap.json"
    )

    print(
        "\n--- RESULT ---"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()