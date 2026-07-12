# Medical Conversation Analysis — Project Summary

## What This Project Does

Takes a doctor-patient conversation (audio or text), converts it to text if needed,
extracts structured clinical information using an LLM, stores the results in a CSV,
and presents everything through a simple web frontend where the user can also download
a medical report.

---

## Pipeline

```
Audio file
    ↓
Whisper (STT)                     ← Week 2
    ↓  transcript text
Single LLM call                   ← Week 1 (extraction + summary in one prompt)
    ↓  JSON string (response.text)
json.loads()
    ↓  nested Python dict
pd.json_normalize(dict, sep='_')  ← flattens nested SOAP sections into columns
    ↓  flat single-row DataFrame (columns still have list values)
pipe-join list columns            ← "flu | headache | migraine"
    ↓  fully flat DataFrame
.to_csv()
    ↓
Streamlit frontend                ← Week 3
```

---

## Tech Stack

| Component    | Tool                                          |
|-------------|-----------------------------------------------|
| Language     | Python                                        |
| Notebook     | Jupyter (.ipynb) for Weeks 1–2                |
| STT          | OpenAI Whisper                                |
| LLM          | Gemini API (`gemini-2.5-flash-lite`, free)    |
| Data handling| pandas                                        |
| Frontend     | Streamlit                                     |
| Dataset      | `har1/MTS_Dialogue-Clinical_Note` HuggingFace |

---

## Project Structure

```
project/
│
├── notebooks/
│   └── week1_exploration.ipynb   ← scratch work, data exploration, prompt testing
│
├── medical_extraction.py         ← production module (already written, keep safe)
├── stt_pipeline.py               ← Week 2
├── app.py                        ← Week 3 (Streamlit)
│
└── data/
    ├── MTS-Dialog-TrainingSet.csv
    └── MTS-Dialog-ValidationSet.csv
```

---

## Dataset Notes (`har1/MTS_Dialogue-Clinical_Note`)

**The dataset is in long format.** Each row is one *section* of one conversation —
the same dialogue repeats across multiple rows with different section labels.

| Column           | What it contains                            |
|-----------------|---------------------------------------------|
| `ID`             | Conversation identifier                     |
| `dialogue`       | The doctor-patient conversation text        |
| `section_header` | Clinical note section label (CC, PLAN, etc) |
| `section_text`   | The doctor's written notes for that section |

**Section headers are NOT disease names — they are clinical note labels:**

| Header         | Full Name             | What it contains                         |
|---------------|-----------------------|------------------------------------------|
| CC            | Chief Complaint       | Why the patient came in                  |
| GENHX         | General History       | Patient's medical background             |
| PASTMEDICALHX | Past Medical History  | Previous diagnoses, surgeries            |
| MEDICATIONS   | Medications           | What the patient is already taking       |
| ALLERGY       | Allergy               | Known allergies                          |
| FAM/SOCHX     | Family/Social History | Family conditions, lifestyle             |
| ROS           | Review of Systems     | Head-to-toe questioning                  |
| EXAM          | Physical Examination  | Doctor's physical findings               |
| ASSESSMENT    | Assessment            | The diagnosis or clinical impression     |
| PLAN          | Plan                  | Treatment plan, prescriptions, referrals |
| DISPOSITION   | Disposition           | What happens after the visit             |

**Critical loading step:**
```python
# Without this, you process the same dialogue multiple times
unique_convos = train.drop_duplicates(subset='ID')
```

**How to use train vs validation:**
- The train/validation split exists for fine-tuning ML models — NOT relevant here
- You are NOT fine-tuning anything. You are prompt engineering a pre-trained LLM
- `train` → develop and iterate your prompt on these dialogues
- `validation` → informally eyeball outputs on unseen conversations at the end
- Evaluation method: human gut-check, not automated scoring

---

## LLM Setup (Gemini API)

**Correct API call pattern:**
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=gemini_api)

response = client.models.generate_content(
    model='gemini-2.5-flash-lite',
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024
    ),
    contents=build_transcript_prompt(sample_transcript)
)

print(response.text)  # .text extracts the string — not the full response object
```

**Key decisions:**
- `system_instruction` carries the rules + JSON schema — separate from the transcript
- `contents` carries the actual dialogue via `build_transcript_prompt()`
- Do NOT concatenate system prompt and user content into one string
- `max_output_tokens=1024` — typical MTS-Dialog responses use ~400–800 tokens
- Check `response.usage_metadata` to verify you're not hitting the ceiling

**Free tier check:** Always verify at `ai.google.dev/gemini-api/docs/pricing`
before hardcoding any model name. If the Free Tier column shows $0 → free.
If it says "Not available" → paid. `gemini-2.5-flash-lite` is currently free.
Models like `gemini-3.1-flash-lite` shown in the Google AI Studio UI are paid.

---

## JSON Output Schema (SOAP Format)

The LLM outputs a **nested** JSON following the SOAP medical record standard:

```json
{
  "summary": "2-3 sentence clinical summary",
  "encounter_type": "initial_consultation | follow_up | results_review | other",

  "subjective": {
    "chief_complaint": "patient's primary reason for visit",
    "symptoms": ["symptom 1", "symptom 2"],
    "symptom_onset": "3 days ago or null",
    "medical_history": ["condition 1", "condition 2"],
    "current_medications": ["medication 1"],
    "allergies": ["allergy 1"]
  },

  "objective": {
    "physical_exam_findings": ["finding 1"],
    "vital_signs": "BP 120/80 or null"
  },

  "assessment": {
    "diagnosis": "primary diagnosis or null",
    "differential_diagnosis": ["alternative 1"]
  },

  "plan": {
    "prescribed_medications": [
      { "name": "...", "dosage": "...", "frequency": "...", "duration": "..." }
    ],
    "treatment_plan": ["instruction 1"],
    "investigations_ordered": ["test 1"],
    "referrals": ["specialist 1"],
    "follow_up": "timeframe or null"
  },

  "additional_notes": "anything else or null",
  "extraction_confidence": "high | medium | low"
}
```

**SOAP stands for:** Subjective / Objective / Assessment / Plan — the standard
format doctors use to document patient encounters.

**`objective` section will mostly be null/empty for this dataset** — MTS-Dialog
dialogues rarely include physical exam narration. This is expected and correct.

---

## JSON → CSV Conversion (The Correct Sequence)

### Step 1 — Parse the JSON string into a Python dict
```python
import json
response_dict = json.loads(response.text)
```

### Step 2 — Flatten the nested dict with pd.json_normalize()
```python
import pandas as pd
df = pd.json_normalize(response_dict, sep='_')
# Produces columns like: subjective_symptoms, plan_follow_up, assessment_diagnosis
```

**Common mistakes to avoid:**
- `pd.DataFrame.from_dict(response_dict)` → WRONG: creates multiple rows,
  uses nested keys as row indices
- `pd.DataFrame(response_dict)` without list wrapper → WRONG for a single row
- `pd.json_normalize()` is the correct tool for nested JSON flattening

### Step 3 — Pipe-join list columns
After `json_normalize`, list fields still contain actual Python lists.
These must be converted to strings before writing to CSV:
```python
" | ".join(items) if items else None
# e.g. ["flu", "headache"] → "flu | headache"
```
Use `|` not `,` as separator to avoid confusing CSV parsers.

### Step 4 — Write to CSV
```python
df.to_csv('output.csv', index=False)
```

### For batch processing (multiple conversations):
Accumulate a list of flat dicts, then create one DataFrame at the end:
```python
rows = []
for dialogue in conversations:
    result = call_llm(dialogue)
    flat = flatten(result)
    rows.append(flat)

df = pd.DataFrame(rows)
df.to_csv('all_extractions.csv', index=False)
```

---

## Useful Google Search Keywords

| Problem | Search terms |
|--------|-------------|
| Rename/strip column prefixes | `pandas rename columns strip prefix` |
| Join list values in a column | `pandas apply lambda join list column` |
| Save DataFrame to Excel | `pandas save dataframe to excel` |
| Flatten nested JSON/dict | `pandas json_normalize nested dict` |
| Read API key from .env file | `python load environment variable dotenv` |

---

## Whisper (STT) Notes

- Whisper is **speech-to-text only** — transcription, nothing else
- Does NOT summarise, analyse, or understand meaning
- No speaker labels by default — can't tell Doctor from Patient
- LLM can infer speaker roles from context even without labels
- For speaker labelling: use **WhisperX** (wraps Whisper + pyannote.audio)
- Recommendation: skip diarisation now, add as Week 4 enhancement if time allows

**Model sizes:**

| Model     | Use case                          |
|-----------|-----------------------------------|
| tiny/base | Fast iteration during development |
| medium    | Recommended starting point        |
| large-v2  | Best accuracy for final pipeline  |

---

## Four-Week Roadmap

### Week 1 — LLM Extraction Pipeline (Justin coding himself)
1. Set up project folder, virtual environment, install `pandas google-generativeai`
2. Store API key in `.env`, never hardcode it
3. Load CSVs with pandas, explore columns, check `value_counts()` and `nunique()`
4. `drop_duplicates(subset='ID')` to get unique conversations
5. Set up Gemini client, run extraction on a single dialogue
6. Eyeball the JSON output — does it look medically sensible?
7. Implement full conversion: `json.loads()` → `json_normalize()` → pipe-join lists → `.to_csv()`
8. Batch over 10 conversations into one CSV

### Week 2 — Whisper STT Integration
- Install Whisper, run on a sample audio file
- Connect Whisper output as input to the Week 1 extraction pipeline
- End-to-end test: audio file → CSV

### Week 3 — Streamlit Frontend
- Display transcript text
- Display extracted SOAP fields as a readable medical report
- Download button for the CSV/report

### Week 4 — Polish + Optional Enhancements
- Error handling and edge cases
- WhisperX speaker diarisation (if time allows)
- General cleanup, README, documentation

---

## Files Already Written (Do Not Lose)

| File | Contents |
|------|----------|
| `medical_extraction.py` | Full production module: system prompt, Gemini call, JSON parsing, CSV helpers, batch processor |

---

## Teaching Preference

Early weeks (1–2): Justin codes himself. Claude gives step-by-step instructor-style
guidance, conceptual explanations, and small targeted snippets only.

Later weeks (3–4): Justin may hand over coding to Claude directly when time is short.
Claude then provides complete scripts.
