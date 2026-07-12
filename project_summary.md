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
Whisper (STT)         ← Week 2
    ↓  transcript text
Single LLM call       ← Week 1 (extraction + summary in one prompt)
    ↓  JSON string
json.loads()
    ↓  Python dict
Flatten list fields   ← pipe-separate lists e.g. "flu | headache | migraine"
    ↓  flat dict
pd.DataFrame([flat_dict])
    ↓
.to_csv()
    ↓
Streamlit frontend    ← Week 3
```

---

## Tech Stack

| Component       | Tool                              |
|----------------|-----------------------------------|
| Language        | Python                            |
| Notebook        | Jupyter (.ipynb) for Weeks 1–2    |
| STT             | OpenAI Whisper                    |
| LLM             | Gemini API (`gemini-2.5-flash-lite`, free tier) |
| Data handling   | pandas                            |
| Frontend        | Streamlit                         |
| Dataset         | `har1/MTS_Dialogue-Clinical_Note` on HuggingFace |

---

## Project Structure

```
project/
│
├── notebooks/
│   └── week1_exploration.ipynb   ← scratch work, data exploration, prompt testing
│
├── medical_extraction.py         ← production module (already written by Claude)
├── stt_pipeline.py               ← Week 2
├── app.py                        ← Week 3 (Streamlit)
│
└── data/
    ├── MTS-Dialog-TrainingSet.csv
    └── MTS-Dialog-ValidationSet.csv
```

---

## Dataset Notes (`har1/MTS_Dialogue-Clinical_Note`)

**The dataset is in long format.** Each row is one section of one conversation.
The same dialogue repeats across multiple rows. Key columns:

| Column          | What it contains                              |
|----------------|-----------------------------------------------|
| `ID`            | Conversation identifier                       |
| `dialogue`      | The doctor-patient conversation text          |
| `section_header`| Clinical note section label (see below)       |
| `section_text`  | The doctor's written notes for that section   |

**Section headers and what they mean:**

| Header         | Full name              | What it contains                        |
|---------------|------------------------|-----------------------------------------|
| CC            | Chief Complaint        | Why the patient came in                 |
| GENHX         | General History        | Patient's medical background            |
| PASTMEDICALHX | Past Medical History   | Previous diagnoses, surgeries           |
| MEDICATIONS   | Medications            | What the patient is already taking      |
| ALLERGY       | Allergy                | Known allergies                         |
| FAM/SOCHX     | Family/Social History  | Family conditions, lifestyle            |
| ROS           | Review of Systems      | Head-to-toe questioning                 |
| EXAM          | Physical Examination   | Doctor's physical findings              |
| ASSESSMENT    | Assessment             | The diagnosis or clinical impression    |
| PLAN          | Plan                   | Treatment plan, prescriptions, referrals|
| DISPOSITION   | Disposition            | What happens after the visit            |

**Critical:** Use `drop_duplicates(subset='ID')` before processing
to avoid sending the same dialogue to the LLM multiple times.

**How to use train vs validation:**
- These are NOT for fine-tuning the LLM (that's a completely separate process
  requiring model weights, thousands of examples, and GPUs — not applicable here)
- `train` → develop and iterate on your prompt using these dialogues
- `validation` → informally check your outputs look sensible on unseen conversations
- Evaluation is human eyeballing, not automated scoring

---

## LLM Key Decisions

**Why a single LLM call (not two):**
The extraction prompt already includes a `summary` field. No need for a
separate summarisation step — the model does both in one call.

**Why not match the dataset's section_header format:**
The dataset format was designed for fine-tuning a BART model, not for
readable output. Your JSON schema with plain English fields (`diagnosis`,
`treatment_plan`, etc.) is better for the CSV, the frontend, and the LLM.

**Gemini API usage — correct pattern:**
```python
response = client.models.generate_content(
    model='gemini-2.5-flash-lite',
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024
    ),
    contents=build_transcript_prompt(sample_transcript)
)
print(response.text)   # .text extracts just the string
```

**Free tier:** Always verify at `ai.google.dev/gemini-api/docs/pricing`.
The Free Tier column shows $0 for free models. `gemini-2.5-flash-lite` is
currently free. Models shown in the pricing UI (e.g. `gemini-3.1-flash-lite`)
may be paid — always confirm before hardcoding a model name.

**max_output_tokens:** 1024 is the right value for this dataset.
Typical MTS-Dialog responses use ~400–600 tokens. Check `response.usage_metadata`
to verify you're not hitting the ceiling.

---

## JSON Output Schema

```json
{
  "summary": "...",
  "chief_complaint": "...",
  "symptoms": ["...", "..."],
  "symptom_onset": "...",
  "medical_history": ["..."],
  "current_medications": ["..."],
  "allergies": ["..."],
  "diagnosis": "...",
  "differential_diagnosis": ["..."],
  "prescribed_medications": [
    { "name": "...", "dosage": "...", "frequency": "...", "duration": "..." }
  ],
  "treatment_plan": ["..."],
  "investigations_ordered": ["..."],
  "referrals": ["..."],
  "follow_up": "...",
  "additional_notes": "...",
  "encounter_type": "initial_consultation | follow_up | results_review | other",
  "extraction_confidence": "high | medium | low"
}
```

**CSV flattening rules:**
- Scalar fields (strings/nulls) → copy directly
- List-of-strings fields → pipe-join: `"flu | headache | migraine"`
- Empty lists → `None`
- `prescribed_medications` → needs special handling (list of dicts)

---

## Whisper (STT) Notes

- Whisper is **speech-to-text only** — it does not summarise or analyse
- No speaker diarisation by default — it can't label who is Doctor vs Patient
- For diarisation, **WhisperX** wraps Whisper + pyannote.audio in one library
- The LLM can infer speaker roles from context even without labels
- Recommended: skip diarisation in core pipeline, add as Week 4 enhancement if time allows

**Whisper model sizes:**

| Model    | Use case                        |
|----------|---------------------------------|
| tiny/base| Fast iteration during dev       |
| medium   | Recommended starting point      |
| large-v2 | Best accuracy for final pipeline|

---

## Four-Week Roadmap

### Week 1 — LLM Extraction Pipeline (doing yourself)
1. Load the dataset in a notebook
2. Explore structure: `value_counts()`, `nunique()`, `drop_duplicates()`
3. Pick 5–10 unique conversations
4. Set up Gemini API client
5. Run `build_transcript_prompt()` + Gemini call on a single dialogue
6. Eyeball the JSON output — does it look medically sensible?
7. Implement JSON → flatten → DataFrame → CSV yourself
8. Batch over 10 conversations and write all rows to one CSV

### Week 2 — Whisper STT Integration
- Install Whisper, run on a sample audio file
- Connect Whisper output as input to your Week 1 extraction pipeline
- End-to-end test: audio file → CSV

### Week 3 — Streamlit Frontend
- Display transcript text
- Display extracted fields as a readable medical report
- Add download button for the CSV/report

### Week 4 — Polish + Optional Enhancements
- Error handling and edge cases
- WhisperX diarisation (if time allows)
- General cleanup and testing

---

## Teaching Preference

Justin is doing the coding himself in Weeks 1–2. Claude gives:
- Step-by-step instructor-style guidance
- Conceptual explanations
- Small targeted snippets when helpful
- No full scripts unless Justin is short on time

From Week 3–4 onwards, more coding will be handed to Claude directly.
