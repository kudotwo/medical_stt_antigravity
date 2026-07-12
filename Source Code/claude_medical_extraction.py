"""
medical_extraction.py
---------------------
LLM-powered extraction of structured clinical data from
doctor–patient conversation transcripts.

Usage:
    result = extract_from_transcript(transcript_text)
    row    = to_csv_row(result)

Dependencies:
    pip install anthropic pandas
"""

import json
import re
import anthropic
import pandas as pd
from pathlib import Path


# ── 1. PROMPTS ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a clinical documentation assistant trained to read doctor–patient
conversation transcripts and extract structured medical information.

Rules:
- Return ONLY a valid JSON object. No explanation, no preamble, no markdown
  code fences (no ```json). The very first character of your output must be {
  and the last must be }.
- Use null for any field not explicitly mentioned in the transcript.
- Use [] for list fields with no items found.
- Never invent, infer, or assume information that is not clearly stated.
- For medications, capture dosage/frequency/duration exactly as spoken —
  do not normalise or abbreviate.
- extraction_confidence reflects transcript clarity (clear audio, complete
  sentences = high; fragmented or ambiguous = low).

Return this exact JSON schema:

{
  "summary": "<2–3 sentence clinical summary of the full encounter>",

  "chief_complaint": "<patient's primary reason for the visit, in their words>",

  "symptoms": [
    "<symptom 1 — include severity/duration if stated>",
    "<symptom 2>",
    "..."
  ],

  "symptom_onset": "<when symptoms started, e.g. '3 days ago' or null>",

  "medical_history": [
    "<relevant past condition or procedure mentioned>",
    "..."
  ],

  "current_medications": [
    "<medication patient was already taking before this visit>",
    "..."
  ],

  "allergies": [
    "<allergy mentioned>",
    "..."
  ],

  "diagnosis": "<primary diagnosis given by the doctor, or null>",

  "differential_diagnosis": [
    "<alternative diagnosis the doctor considered>",
    "..."
  ],

  "prescribed_medications": [
    {
      "name": "<medication name>",
      "dosage": "<e.g. 500mg or null>",
      "frequency": "<e.g. twice daily or null>",
      "duration": "<e.g. 7 days or null>"
    }
  ],

  "treatment_plan": [
    "<recommendation or instruction given by the doctor>",
    "..."
  ],

  "investigations_ordered": [
    "<test, lab, or imaging ordered>",
    "..."
  ],

  "referrals": [
    "<specialist or department referred to>",
    "..."
  ],

  "follow_up": "<follow-up instructions or timeframe, or null>",

  "additional_notes": "<any other clinically relevant detail not captured above, or null>",

  "extraction_confidence": "high | medium | low"
}
""".strip()


def build_user_prompt(transcript: str) -> str:
    """Wrap a transcript in the extraction prompt."""
    return f"""
    Extract all clinical information from the doctor-patient conversation below.
    Follow the JSON schema exactly as specified.

    <transcript>
    {transcript.strip()}
    </transcript>
    """.strip()


# ── 2. LLM CALL ──────────────────────────────────────────────────────────────

def extract_from_transcript(
    transcript: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1500,
) -> dict:
    """
    Send a transcript to Claude and return a parsed extraction dict.

    Args:
        transcript: Raw text of the doctor–patient conversation.
        model:      Anthropic model to use.
        max_tokens: Upper limit for the response.

    Returns:
        Parsed dict matching the extraction schema.
        On failure, returns a dict with an 'error' key.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(transcript)}
        ],
    )

    raw = message.content[0].text
    return _parse_json_response(raw)


def _parse_json_response(raw: str) -> dict:
    """
    Safely parse the LLM response to JSON.
    Strips accidental markdown fences if the model adds them despite instructions.
    """
    # Remove ```json ... ``` or ``` ... ``` wrappers if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse failed: {e}",
            "raw_response": raw,
        }


# ── 3. CSV HELPERS ───────────────────────────────────────────────────────────

# Fields that are lists of plain strings → joined with " | " in the CSV
_LIST_FIELDS = [
    "symptoms",
    "medical_history",
    "current_medications",
    "allergies",
    "differential_diagnosis",
    "treatment_plan",
    "investigations_ordered",
    "referrals",
]

# Fields that are lists of objects (prescribed_medications) → serialised as JSON
_OBJECT_LIST_FIELDS = ["prescribed_medications"]


def to_csv_row(extracted: dict, conversation_id: str = "") -> dict:
    """
    Flatten an extraction dict into a single CSV-ready row.

    List fields become pipe-separated strings: "headache | fever | fatigue"
    Medication objects are stored as a compact JSON string.

    Args:
        extracted:       Output of extract_from_transcript().
        conversation_id: Optional identifier for the source conversation.

    Returns:
        Flat dict suitable for pandas.DataFrame or csv.DictWriter.
    """
    if "error" in extracted:
        return {
            "conversation_id": conversation_id,
            "error": extracted["error"],
            "raw_response": extracted.get("raw_response", ""),
        }

    row = {"conversation_id": conversation_id}

    # Scalar fields — copy directly
    scalar_fields = [
        "summary", "chief_complaint", "symptom_onset",
        "diagnosis", "follow_up", "additional_notes",
        "extraction_confidence",
    ]
    for field in scalar_fields:
        row[field] = extracted.get(field)

    # List-of-strings fields → pipe-joined
    for field in _LIST_FIELDS:
        items = extracted.get(field) or []
        row[field] = " | ".join(str(i) for i in items) if items else None

    # List-of-objects fields → JSON string
    for field in _OBJECT_LIST_FIELDS:
        items = extracted.get(field) or []
        row[field] = json.dumps(items, ensure_ascii=False) if items else None

    return row


# ── 4. BATCH PROCESSING ──────────────────────────────────────────────────────

def process_dataset(
    transcripts: list[dict],
    output_csv: str = "medical_extractions.csv",
) -> pd.DataFrame:
    """
    Run extraction over a list of transcript dicts and save to CSV.

    Args:
        transcripts: List of {"id": str, "text": str} dicts.
        output_csv:  Path to write the resulting CSV.

    Returns:
        DataFrame of all extracted rows.
    """
    rows = []

    for item in transcripts:
        conv_id = item.get("id", "")
        text    = item.get("text", "")
        print(f"Processing: {conv_id} ...")

        extracted = extract_from_transcript(text)
        row       = to_csv_row(extracted, conversation_id=conv_id)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows → {output_csv}")
    return df


# ── 5. QUICK TEST ─────────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT = """
Doctor: Good morning. What brings you in today?
Patient: Hi doctor. I've had this really bad headache for about three days now.
         It's mostly on the right side and it's throbbing. Light really bothers
         me and I've been feeling a bit nauseous too.
Doctor: I see. Have you had headaches like this before?
Patient: Yes, I get them occasionally, maybe once or twice a year, but this one
         is worse than usual.
Doctor: Any fever, neck stiffness, or changes in vision?
Patient: No fever. My neck feels fine. Vision is normal.
Doctor: Any medications you're currently taking?
Patient: Just a daily multivitamin. I tried ibuprofen two days ago but it barely
         helped.
Doctor: Any known drug allergies?
Patient: No allergies.
Doctor: Based on what you're describing — unilateral throbbing headache,
        photophobia, nausea, and a prior history — this sounds very consistent
        with migraine. I'm going to prescribe sumatriptan 50mg. Take it at the
        onset of the next headache. If you don't get relief in two hours you can
        take a second dose. I'd also like you to keep a headache diary to track
        frequency and triggers. Let's also order a CBC to rule out anything else.
Patient: Okay, thank you.
Doctor: Come back in four weeks, or sooner if the headaches get more frequent
        or you develop new symptoms.
"""


if __name__ == "__main__":
    print("Running extraction on sample transcript...\n")
    result = extract_from_transcript(SAMPLE_TRANSCRIPT)

    if "error" in result:
        print("Extraction failed:", result["error"])
    else:
        print("── Extracted JSON ──────────────────────────────")
        print(json.dumps(result, indent=2))

        print("\n── CSV Row ─────────────────────────────────────")
        row = to_csv_row(result, conversation_id="sample_001")
        for key, value in row.items():
            print(f"  {key:<30} {value}")
