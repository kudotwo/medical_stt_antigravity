"""
evaluate_models.py
==================
Evaluates three LLM models on SOAP extraction quality, scored by an LLM-as-a-judge.

Models compared (SOAP extraction):
  - gemini-3.1-flash-lite  (Google GenAI)
  - gemini-3.5-flash        (Google GenAI)
  - gpt-5.6-terra           (OpenAI — best thinking-to-cost ratio)

Judge: gemini-3.5-flash (Gemini stays as judge, no extra OpenAI tokens spent)

Scores per (case, model):
  1. Clinical Fidelity    (1-10)
  2. Logical Consistency  (1-10)
  3. Predictive Accuracy  (1-10)

Loop order — alternating / conversation-first:
  CAR0001 x [gemini-3.1-flash-lite, gemini-3.5-flash, gpt-5.6-terra]
  MSK0005 x [...]  ...
  If the API cuts out mid-run, partial runs still have complete data
  for finished conversations across ALL three models.

Checkpoint system:
  Each (case, model) pair saves to metrics_output/{case}_{safe_model}.json.
  Existing checkpoints are skipped — fully resumable.

Output:
  - metrics_output/{case}_{model}.json  (per-pair checkpoint)
  - model_comparison_metrics.csv        (combined 15-row table, 5 cases x 3 models)
"""

import os
import sys
import json
import csv
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

BASE_DIR         = Path(__file__).parent
SOURCE_DIR       = BASE_DIR.parent
AUDIO_DIR        = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Audio_Recordings'
TRANSCRIPT_DIR   = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Clean_Transcripts'
PREDICTED_DIR    = BASE_DIR / 'predicted'
GROUND_TRUTH_DIR = BASE_DIR / 'ground_truth'
METRICS_CSV      = BASE_DIR / 'model_comparison_metrics.csv'

# Add Source Code to sys.path so we can import stt_pipeline_gpu
sys.path.insert(0, str(SOURCE_DIR))
import stt_pipeline_gpu

# ── Model configuration ────────────────────────────────────────────────────────
MODELS_TO_EVALUATE = [
    'gemini-3.1-flash-lite',  # Gemini (Google GenAI)
    'gemini-3.5-flash',       # Gemini (Google GenAI)
    'gpt-5.6-terra',          # OpenAI — best thinking-to-cost ratio for medical SOAP
]

CASES = ['CAR0001', 'MSK0005', 'RES0050', 'GAS0003', 'DER0001']

# Judge stays on Gemini so we don't spend OpenAI tokens on grading
JUDGE_MODEL = 'gemini-3.5-flash'


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_segments(case_id):
    """Load pre-computed Whisper segments from cache JSON."""
    cache_file = AUDIO_DIR / f"{case_id}_segments.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing segment cache: {cache_file}")
    with open(cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ground_truth_transcript(case_id):
    txt_path = TRANSCRIPT_DIR / f"{case_id}.txt"
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding='utf-8')


def get_ground_truth_soap(case_id):
    csv_path = GROUND_TRUTH_DIR / f"{case_id}_soap.csv"
    if not csv_path.exists():
        return ""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return json.dumps(rows[0], indent=2) if rows else ""


# ── LLM-as-a-judge (Gemini) ───────────────────────────────────────────────────

# Primary and fallback judge models (in order of preference)
_JUDGE_MODELS = ['gemini-3.5-flash', 'gemini-3.1-flash-lite']


def _parse_judge_response(raw_text):
    """Strip markdown fences and parse JSON from a judge response."""
    text = raw_text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith('`'):
        lines = text.split('\n')
        # Drop first line (```json or ```) and last line (```)
        text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
    text = text.strip()
    return json.loads(text)


def evaluate_soap_with_judge(gt_transcript, gt_soap, predicted_soap_dict):
    """
    Grade a predicted SOAP using Gemini as judge.
    - Tries gemini-3.5-flash first, falls back to gemini-3.1-flash-lite.
    - Up to 10 attempts total with exponential back-off (30s → 60s → 120s → 120s...).
    - Strips markdown fences before JSON parsing to handle partial/malformed responses.
    Returns {"clinical_fidelity": int, "logical_consistency": int, "predictive_accuracy": int}.
    """
    client = genai.Client()
    predicted_soap_json = json.dumps(predicted_soap_dict, indent=2)

    prompt = f"""You are an expert medical transcription judge. Evaluate the predicted SOAP report based on the Ground Truth Transcript and Ground Truth SOAP report.
You must provide a score from 1 to 10 for each of the following metrics:
1. Clinical Fidelity: How accurately does the predicted SOAP capture the clinical facts from the Ground Truth Transcript without omissions or hallucinations?
2. Logical Consistency: Are the Plan and Diagnosis logically supported by the Subjective and Objective findings in the predicted SOAP?
3. Predictive Accuracy: How closely does the predicted SOAP match the structure, detail, and semantic meaning of the Ground Truth SOAP?

Ground Truth Transcript:
{gt_transcript}

Ground Truth SOAP:
{gt_soap}

Predicted SOAP:
{predicted_soap_json}

Output MUST be a valid JSON object strictly matching this schema:
{{
  "clinical_fidelity": <int 1-10>,
  "logical_consistency": <int 1-10>,
  "predictive_accuracy": <int 1-10>
}}
"""
    max_attempts = 10
    for attempt in range(max_attempts):
        # Rotate to fallback judge model after half the attempts
        judge = _JUDGE_MODELS[0] if attempt < max_attempts // 2 else _JUDGE_MODELS[-1]
        try:
            response = client.models.generate_content(
                model=judge,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            result = _parse_judge_response(response.text)
            if attempt > 0:
                print(f"    [Judge] Succeeded on attempt {attempt+1} using {judge}")
            return result
        except Exception as e:
            wait = min(30 * (2 ** attempt), 120)  # 30, 60, 120, 120, ...
            print(f"    [Judge] Error (attempt {attempt+1}/{max_attempts}, model={judge}): {e}")
            print(f"    [Judge] Waiting {wait}s before retry...")
            time.sleep(wait)

    print("    [Judge] Failed after all retries — returning zeros.")
    return {"clinical_fidelity": 0, "logical_consistency": 0, "predictive_accuracy": 0}


# ── SOAP extraction routing ───────────────────────────────────────────────────

def _extract_soap_gemini(model, segments):
    """
    Extract SOAP using a Gemini model directly.
    Mirrors the original Gemini path from stt_pipeline_gpu without touching GPT_MODEL.
    """
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=GEMINI_API_KEY)
    user_msg = ('Label each segment as Doctor or Patient and extract SOAP.\n\nTRANSCRIPT:\n'
                + stt_pipeline_gpu._fmt(segments))
    print(f'  [SOAP] Gemini ({model}) diarization + SOAP extraction...')
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[
                {'role': 'user',  'parts': [{'text': stt_pipeline_gpu.SYSTEM_PROMPT}]},
                {'role': 'model', 'parts': [{'text': 'Understood. Producing SOAP JSON.'}]},
                {'role': 'user',  'parts': [{'text': user_msg}]},
            ]
        )
        raw = resp.text.strip()
        if raw.startswith('`'):
            raw = raw.split('`')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        raw = raw.strip()
        print(f'  RAW (first 200 chars): {raw[:200]}')
        return json.loads(raw)
    except Exception as e:
        print(f'  [ERROR] Gemini SOAP extraction failed: {e}')
        traceback.print_exc()
        return None


def extract_soap(model, segments):
    """
    Route SOAP extraction to the correct provider:
      gpt-*    -> OpenAI via stt_pipeline_gpu (GPT_MODEL patched)
      gemini-* -> Gemini GenAI (direct call, leaves pipeline's GPT_MODEL alone)
    """
    if model.startswith('gpt-'):
        stt_pipeline_gpu.GPT_MODEL = model
        return stt_pipeline_gpu.diarize_and_extract_soap(segments)
    else:
        return _extract_soap_gemini(model, segments)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    import pandas as pd

    METRICS_DIR = BASE_DIR / 'metrics_output'
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Medical STT — 3-Model SOAP Evaluation")
    print(f"  Models : {MODELS_TO_EVALUATE}")
    print(f"  Cases  : {CASES}")
    print(f"  Judge  : {JUDGE_MODEL}")
    print("=" * 65)

    # Outer loop: conversation (ensures alternating pattern)
    for case_id in CASES:
        print(f"\n{'─'*60}")
        print(f"  Conversation: {case_id}")
        print(f"{'─'*60}")

        try:
            segments     = get_segments(case_id)
            gt_transcript = get_ground_truth_transcript(case_id)
            gt_soap      = get_ground_truth_soap(case_id)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # Inner loop: model
        for model in MODELS_TO_EVALUATE:
            print(f"\n  Model: {model}")

            # Sanitise model name for filenames (avoids issues with dots/slashes)
            safe_model  = model.replace('/', '_').replace(':', '_')
            metric_file = METRICS_DIR / f"{case_id}_{safe_model}.json"

            if metric_file.exists():
                print(f"    ✅ Checkpoint exists — skipping")
                continue

            # Step 1: SOAP extraction
            print(f"    → Extracting SOAP...")
            predicted_soap = extract_soap(model, segments)

            if not predicted_soap:
                print(f"\n  [ERROR] SOAP extraction returned None for {case_id} / {model}.")
                print("  Fix the error above then re-run — existing checkpoints will be skipped.")
                sys.exit(1)

            # Step 2: Save predicted SOAP CSV + diarized transcript
            model_out_dir = PREDICTED_DIR / safe_model
            model_out_dir.mkdir(parents=True, exist_ok=True)

            soap_only = {k: v for k, v in predicted_soap.items() if k != 'diarized_segments'}
            flat_soap = stt_pipeline_gpu._flatten(pd.json_normalize(soap_only, sep='.'))
            flat_soap['audio_file'] = f"{case_id}.mp3"
            flat_soap.to_csv(model_out_dir / f"{case_id}_soap.csv", index=False)

            segs     = predicted_soap.get('diarized_segments', [])
            diarized = '\n'.join([
                f'[{s.get("start", 0):.1f}s->{s.get("end", 0):.1f}s] '
                f'{s.get("speaker", "?")}: {s.get("text", "").strip()}'
                for s in segs
            ])
            (model_out_dir / f"{case_id}_transcript.txt").write_text(diarized, encoding='utf-8')
            print(f"    → Saved to predicted/{safe_model}/")

            # Step 3: Judge evaluation
            print(f"    → Grading with {JUDGE_MODEL}...")
            scores = evaluate_soap_with_judge(gt_transcript, gt_soap, soap_only)

            if scores.get('clinical_fidelity') == 0 and scores.get('logical_consistency') == 0:
                print(f"\n  [ERROR] Judge returned all-zeros — likely an API failure.")
                print("  Re-run to resume; this checkpoint will NOT be written.")
                sys.exit(1)

            print(
                f"    ✅ Fidelity: {scores['clinical_fidelity']} | "
                f"Consistency: {scores['logical_consistency']} | "
                f"Accuracy: {scores['predictive_accuracy']}"
            )

            # Step 4: Write checkpoint JSON
            result_data = {
                'case_id': case_id,
                'model':   model,
                'clinical_fidelity':   scores.get('clinical_fidelity', 0),
                'logical_consistency': scores.get('logical_consistency', 0),
                'predictive_accuracy': scores.get('predictive_accuracy', 0),
            }
            with open(metric_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)
            print(f"    ✅ Checkpoint saved: {metric_file.name}")

    print(f"\n{'='*65}")
    print("  ✅ All cases processed!")
    print(f"{'='*65}")

    # Combine all checkpoint JSONs into a single CSV
    combined_results = []
    for case_id in CASES:
        for model in MODELS_TO_EVALUATE:
            safe_model  = model.replace('/', '_').replace(':', '_')
            metric_file = METRICS_DIR / f"{case_id}_{safe_model}.json"
            if metric_file.exists():
                with open(metric_file, 'r', encoding='utf-8') as f:
                    combined_results.append(json.load(f))

    with open(METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'case_id', 'model',
            'clinical_fidelity', 'logical_consistency', 'predictive_accuracy'
        ])
        writer.writeheader()
        writer.writerows(combined_results)

    total_expected = len(CASES) * len(MODELS_TO_EVALUATE)
    print(f"\n  ✅ Combined metrics → {METRICS_CSV}")
    print(f"     Rows: {len(combined_results)} / {total_expected} expected")


if __name__ == '__main__':
    main()
