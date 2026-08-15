"""
stt_pipeline_phi.py
Local experiment: existing STT JSON -> Phi-4-mini extraction -> Phi validation -> SOAP.
No audio transcription in this version.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


EXTRACTION_PROMPT = r"""
You are a strict medical information extraction system.

Extract ONLY information explicitly present in the doctor-patient transcript.

Rules:
- Do not invent, infer, assume, or use outside medical knowledge.
- Doctor questions are NOT patient symptoms or negatives.
- An unanswered question is NOT an answer and does not imply "No".
- For symptoms, complaints, medications, allergies, history, and lifestyle, use
  patient statements or explicit patient confirmations.
- For diagnoses, tests, treatment, referrals, and follow-up, include them only
  when explicitly stated by the doctor/patient.
- Preserve wording as much as practical; do not add anatomical details.
- If information is absent, use null or [].
- Return ONLY valid JSON.

Use exactly this structure:

{
  "patient_demographics": {"age": null, "gender": null},
  "chief_complaint": null,
  "symptoms": [
    {
      "description": null,
      "location": null,
      "character": null,
      "onset": null,
      "change_over_time": null,
      "aggravating_factors": [],
      "relieving_factors": []
    }
  ],
  "associated_symptoms": [],
  "pertinent_negatives": [],
  "current_medications": [],
  "allergies": [],
  "medical_history": [],
  "additional_notes": [],
  "diagnoses": [],
  "investigations_ordered": [],
  "treatment_plan": [],
  "referrals": [],
  "follow_up": null
}
"""


VALIDATION_PROMPT = r"""
You are a strict evidence validator for medical documentation.

Determine whether ONE extracted fact is explicitly supported by the transcript.

Rules:
- The transcript is the only source of truth.
- Doctor questions do NOT support a patient symptom or negative.
- No patient answer means UNSUPPORTED.
- Do not infer or use medical knowledge.
- If any part of the fact is unsupported, return UNSUPPORTED.
- Return ONLY one word: SUPPORTED or UNSUPPORTED.
"""


def ollama_chat(system: str, user: str, json_mode: bool = False) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
    }
    if json_mode:
        payload["format"] = "json"

    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    text = r.json().get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text


def load_segments(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Segments file must contain a non-empty JSON list.")
    return data


def format_transcript(segments: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'Segment {s.get("id", i)} '
        f'[{float(s.get("start", 0)):.1f}s->{float(s.get("end", 0)):.1f}s]: '
        f'{str(s.get("text", "")).strip()}'
        for i, s in enumerate(segments)
    )


def parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object.")
    return result


def extract_facts(transcript: str) -> dict[str, Any]:
    print(f"[Stage 1] Extracting facts with {OLLAMA_MODEL}...")
    start = time.time()
    raw = ollama_chat(
        EXTRACTION_PROMPT,
        f"TRANSCRIPT:\n{transcript}",
        json_mode=True,
    )
    print(f"[Stage 1] {time.time() - start:.1f}s")
    return parse_json(raw)


def flatten_facts(obj: Any, path: tuple = ()) -> list[tuple[tuple, Any]]:
    facts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            facts.extend(flatten_facts(v, path + (k,)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            facts.extend(flatten_facts(v, path + (i,)))
    elif obj is not None and obj != "":
        facts.append((path, obj))
    return facts


def path_str(path: tuple) -> str:
    out = ""
    for p in path:
        out += f"[{p}]" if isinstance(p, int) else (("." if out else "") + str(p))
    return out


def validate_fact(transcript: str, path: tuple, value: Any) -> bool:
    fact = json.dumps(value, ensure_ascii=False)
    question = (
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"EXTRACTED FACT:\nField: {path_str(path)}\nValue: {fact}\n\n"
        "Is this exact fact explicitly supported by the transcript?"
    )
    raw = ollama_chat(VALIDATION_PROMPT, question).upper()
    return "SUPPORTED" in raw and "UNSUPPORTED" not in raw


def delete_path(root: Any, path: tuple) -> None:
    if not path:
        return
    parent = root
    for p in path[:-1]:
        parent = parent[p]
    last = path[-1]
    if isinstance(parent, list) and isinstance(last, int):
        parent.pop(last)
    elif isinstance(parent, dict):
        parent[last] = None


def remove_invalid(
    extracted: dict[str, Any],
    invalid: list[tuple],
) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(extracted, ensure_ascii=False))
    # Remove list items from deepest/highest indexes first.
    for path in sorted(
        invalid,
        key=lambda p: (
            len(p),
            p[:-1],
            p[-1] if isinstance(p[-1], int) else -1,
        ),
        reverse=True,
    ):
        delete_path(cleaned, path)
    return cleaned


def build_soap(validated: dict[str, Any]) -> dict[str, Any]:
    symptoms = validated.get("symptoms", [])
    onset = None
    if isinstance(symptoms, list):
        onset_values = [
            s.get("onset")
            for s in symptoms
            if isinstance(s, dict) and s.get("onset")
        ]
        onset = "; ".join(onset_values) if onset_values else None

    return {
        "diarized_segments": [],  # filled from original segments by main()
        "summary": None,
        "encounter_type": "initial_consultation",
        "additional_notes": validated.get("additional_notes", []),
        "extraction_confidence": None,
        "subjective": {
            "chief_complaint": validated.get("chief_complaint"),
            "symptoms": symptoms,
            "symptom_onset": onset,
            "medical_history": validated.get("medical_history", []),
            "current_medications": validated.get("current_medications", []),
            "allergies": validated.get("allergies", []),
        },
        "objective": {
            "physical_exam_findings": [],
            "vital_signs": None,
        },
        "assessment": {
            "diagnosis": validated.get("diagnoses", []),
            "differential_diagnosis": [],
        },
        "plan": {
            "prescribed_medications": [],
            "treatment_plan": validated.get("treatment_plan", []),
            "investigations_ordered": validated.get("investigations_ordered", []),
            "referrals": validated.get("referrals", []),
            "follow_up": validated.get("follow_up"),
        },
    }


def run(segments_file: str, output_file: str) -> None:
    start = time.time()

    print("=" * 60)
    print("Phi-4-mini: extraction -> validation -> SOAP")
    print("=" * 60)

    segments = load_segments(segments_file)
    transcript = format_transcript(segments)
    print(f"[Input] Loaded {len(segments)} segments")

    # Stage 1: extract structured facts.
    extracted = extract_facts(transcript)
    print("\n[Stage 1] Extracted:")
    print(json.dumps(extracted, indent=2, ensure_ascii=False))

    # Stage 2: validate every leaf fact independently.
    facts = flatten_facts(extracted)
    print(f"\n[Stage 2] Validating {len(facts)} facts...")

    invalid = []
    validation_log = []

    for i, (path, value) in enumerate(facts, 1):
        print(f"  [{i}/{len(facts)}] {path_str(path)} = {value!r}")
        try:
            ok = validate_fact(transcript, path, value)
        except Exception as exc:
            print(f"       validator error: {exc}")
            ok = False  # fail closed
        print(f"       -> {'SUPPORTED' if ok else 'UNSUPPORTED'}")
        validation_log.append({
            "field": path_str(path),
            "value": value,
            "supported": ok,
        })
        if not ok:
            invalid.append(path)

    validated = remove_invalid(extracted, invalid)

    soap = build_soap(validated)
    soap["diarized_segments"] = [
        {
            "id": s.get("id"),
            "speaker": None,
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("text", ""),
        }
        for s in segments
    ]

    output = {
        "model": OLLAMA_MODEL,
        "invalid_fact_count": len(invalid),
        "validated_extraction": validated,
        "soap": soap,
        "validation_log": validation_log,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Output] {output_file}")
    print(f"[Total] {time.time() - start:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "segments_file",
        nargs="?",
        default="audio_sample/audio_recordings/Audio_Recordings/CAR0001_segments.json",
    )
    parser.add_argument(
        "--output",
        default="car0001_phi_soap.json",
    )
    args = parser.parse_args()

    try:
        run(args.segments_file, args.output)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
