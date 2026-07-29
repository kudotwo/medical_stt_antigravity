# Batch SOAP Analysis Report
**Generated:** 2026-07-22
**Pipeline:** Whisper Large-v3 (faster-whisper) → Gemini 2.0 Flash SOAP Extraction

---

## 1. Run Overview & Errors

### Batch Run Summary
- **Total predicted SOAP reports generated:** 271
- **Total ground truth SOAP reports available:** 224
- **Matched cases (both predicted & ground truth exist):** 223
- **Predicted-only (no matching ground truth):** 48
- **Ground truth-only (no matching predicted):** 1

### Category Breakdown

| Category | Matched | Predicted Only | Ground Truth Only |
|----------|---------|----------------|-------------------|
| CAR | 5 | 0 | 0 |
| DER | 1 | 0 | 0 |
| GAS | 6 | 0 | 0 |
| GEN | 1 | 0 | 0 |
| MSK | 46 | 0 | 0 |
| RES | 164 | 48 | 1 |

### Errors Encountered During Batch Processing

Two error log files were generated:

#### Error Log 1: `errors_20260721_235647.txt` (25 KB)

**Error Type:** `RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`

**Stage:** Transcription (Whisper model inference)

**Affected Files (12 files):**
- `CAR0004.mp3`
- `CAR0005.mp3`
- `DER0001.mp3`
- `GAS0001.mp3`
- `GAS0002.mp3`
- `GAS0003.mp3`
- `GAS0004.mp3`
- `GAS0005.mp3`
- `GAS0007.mp3`
- `GEN0001.mp3`
- `MSK0001.mp3`
- `MSK0003.mp3`

**Root Cause:** The CUDA cuBLAS library was not properly loaded when the batch script started. This is a GPU driver/CUDA toolkit environment issue — the `LD_LIBRARY_PATH` likely did not include the path to `libcublas.so.12` at the time of script launch. These files were processed early in the run before the environment was corrected (or a restart occurred with the second log).

**Impact:** These 12 files failed at the transcription stage. However, the script appears to have been re-run (second error log timestamp `000416`), and most of these files *do* have predicted SOAP reports, suggesting they were successfully processed on a subsequent run with the CUDA issue resolved.

#### Error Log 2: `errors_20260722_000416.txt` (1.3 KB)

**Error Type:** `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte`

**Stage:** Ground truth transcript reading

**Affected Files (2 files):**
- `RES0002.txt`
- `RES0054.txt`

**Root Cause:** These transcript files appear to be encoded in UTF-16 (BOM `0xFF 0xFE`) rather than UTF-8. The batch script attempted to read them with `encoding='utf-8'`, which failed.

**Impact:** Ground truth SOAP reports could not be generated for RES0002 and RES0054. The predicted reports for these cases (from audio) were still generated successfully. This means RES0002 and RES0054 cannot be compared.

### Cases With Predicted Reports But No Ground Truth

**48 cases:** These have predicted SOAP notes from audio but no corresponding ground truth transcript was processed.

`RES0002`, `RES0054`, `RES0169`, `RES0171`, `RES0172`, `RES0173`, `RES0175`, `RES0176`, `RES0177`, `RES0178`, `RES0179`, `RES0180`, `RES0181`, `RES0182`, `RES0183`, `RES0184`, `RES0185`, `RES0186`, `RES0187`, `RES0188`, `RES0189`, `RES0190`, `RES0191`, `RES0192`, `RES0193`, `RES0194`, `RES0195`, `RES0196`, `RES0197`, `RES0198`, `RES0199`, `RES0200`, `RES0201`, `RES0202`, `RES0203`, `RES0204`, `RES0205`, `RES0206`, `RES0207`, `RES0208`, `RES0209`, `RES0210`, `RES0211`, `RES0212`, `RES0213`, `RES0215`, `RES0216`, `RES0217`

### Cases With Ground Truth But No Predicted Report

**1 cases:** Ground truth was generated from the transcript, but the audio pipeline didn't produce a predicted SOAP. This likely means the batch run was stopped (Gemini token quota exhausted) before reaching these files.

`RES0005`

---

## 2. Field-Level Comparison (Predicted vs Ground Truth)

Analysis across **223 matched cases**.

### Average Text Similarity Per SOAP Field

| SOAP Field | Avg Similarity | Pred Populated | GT Populated | Both Populated |
|------------|----------------|----------------|--------------|----------------|
| `subjective.chief_complaint` | 79.08% | 223/223 (100%) | 223/223 (100%) | 223/223 (100%) |
| `subjective.symptoms` | 70.03% | 223/223 (100%) | 223/223 (100%) | 223/223 (100%) |
| `subjective.symptom_onset` | 77.68% | 223/223 (100%) | 223/223 (100%) | 223/223 (100%) |
| `subjective.medical_history` | 73.52% | 189/223 (85%) | 187/223 (84%) | 183/223 (82%) |
| `subjective.current_medications` | 82.17% | 162/223 (73%) | 162/223 (73%) | 157/223 (70%) |
| `subjective.allergies` | 65.96% | 102/223 (46%) | 92/223 (41%) | 81/223 (36%) |
| `objective.physical_exam_findings` | 43.95% | 207/223 (93%) | 212/223 (95%) | 203/223 (91%) |
| `objective.vital_signs` | 99.55% | 0/223 (0%) | 1/223 (0%) | 0/223 (0%) |
| `assessment.diagnosis` | 75.05% | 111/223 (50%) | 114/223 (51%) | 95/223 (43%) |
| `assessment.differential_diagnosis` | 73.20% | 207/223 (93%) | 204/223 (91%) | 200/223 (90%) |
| `plan.prescribed_medications` | 89.88% | 47/223 (21%) | 44/223 (20%) | 38/223 (17%) |
| `plan.treatment_plan` | 60.23% | 210/223 (94%) | 210/223 (94%) | 209/223 (94%) |
| `plan.investigations_ordered` | 83.43% | 180/223 (81%) | 178/223 (80%) | 175/223 (78%) |
| `plan.referrals` | 97.23% | 23/223 (10%) | 23/223 (10%) | 21/223 (9%) |
| `plan.follow_up` | 55.91% | 203/223 (91%) | 203/223 (91%) | 198/223 (89%) |

### Per-Section Average Similarity

| Section | Avg Similarity |
|---------|----------------|
| Subjective | 74.74% |
| Objective | 71.75% |
| Assessment | 74.12% |
| Plan | 77.33% |
| **Overall** | **75.12%** |

### Encounter Type Agreement

- Matching encounter type: **223/223** (100%)

### Extraction Confidence Distribution

| Confidence | Predicted Count | Ground Truth Count |
|------------|-----------------|-------------------|
| high | 223 | 223 |

---

## 3. Per-Case Similarity Scores

### Top 10 Most Similar Cases

| Case ID | Avg Similarity |
|---------|----------------|
| RES0076 | 91.92% |
| CAR0002 | 90.50% |
| MSK0050 | 89.34% |
| RES0111 | 89.18% |
| MSK0037 | 88.47% |
| MSK0049 | 87.94% |
| CAR0001 | 87.49% |
| RES0038 | 87.45% |
| RES0029 | 87.45% |
| RES0161 | 87.44% |

### Bottom 10 Least Similar Cases

| Case ID | Avg Similarity |
|---------|----------------|
| MSK0021 | 60.87% |
| RES0144 | 58.32% |
| MSK0040 | 57.49% |
| RES0089 | 56.62% |
| RES0053 | 56.15% |
| MSK0044 | 53.34% |
| RES0123 | 52.85% |
| RES0049 | 49.04% |
| MSK0034 | 47.81% |
| RES0147 | 43.73% |

### Similarity Score Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 90-100% | 2 | 1% |
| 70-90% | 169 | 76% |
| 50-70% | 49 | 22% |
| 30-50% | 3 | 1% |
| 0-30% | 0 | 0% |

---

## 4. Detailed Sample Comparisons

Below are side-by-side comparisons for representative cases from each specialty category.

### Case: CAR0001

**Summary Similarity:** 22.55%

**Predicted Summary:** The patient is a 39-year-old male presenting with acute left-sided chest pain that began 8 hours ago. He describes the pain as sharp, rated 7-8/10, exacerbated by lying down and deep breathing, and associated with lightheadedness, dyspnea, and palpitations.

**Ground Truth Summary:** The patient, a 39-year-old male, presents with 8 hours of sharp, constant left-sided chest pain that worsens with lying down and deep breaths. He reports associated symptoms of lightheadedness, dyspnea, and mild palpitations, and noted visible neck swelling without accompanying pain.

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 100.00% | 1 | 1 | ✅ |
| symptoms | 69.26% | 8 | 7 | ⚠️ |
| symptom_onset | 100.00% | 1 | 1 | ✅ |
| medical_history | 100.00% | 0 | 0 | ✅ |
| current_medications | 100.00% | 0 | 0 | ✅ |
| allergies | 100.00% | 1 | 1 | ✅ |
| physical_exam_findings | 43.14% | 1 | 2 | ⚠️ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 0.00% | 0 | 4 | ❌ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 100.00% | 0 | 0 | ✅ |
| investigations_ordered | 100.00% | 0 | 0 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 100.00% | 0 | 0 | ✅ |

**Chief Complaint:**
- Predicted: `chest pain`
- Ground Truth: `Chest pain`

---

### Case: CAR0003

**Summary Similarity:** 27.60%

**Predicted Summary:** Patient presented with a two-month history of progressive exertional breathlessness, orthopnea, and lower extremity swelling. Given the history of prior myocardial infarction, current hypertension, hyperlipidemia, and clinical presentation of frothy sputum and positional dyspnea, the doctor is evalu...

**Ground Truth Summary:** The patient, with a history of heart attack, hypertension, diabetes, and hyperlipidemia, presents with a two-month history of progressive exertional dyspnea, orthopnea, and lower extremity edema. The doctor is investigating for congestive heart failure and plans to order diagnostic tests to evaluate...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 44.07% | 1 | 1 | ⚠️ |
| symptoms | 88.09% | 7 | 7 | ✅ |
| symptom_onset | 84.62% | 1 | 1 | ✅ |
| medical_history | 84.26% | 4 | 5 | ✅ |
| current_medications | 66.67% | 3 | 3 | ⚠️ |
| allergies | 100.00% | 0 | 0 | ✅ |
| physical_exam_findings | 27.59% | 1 | 1 | ❌ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 1 | 1 | ✅ |
| differential_diagnosis | 100.00% | 0 | 0 | ✅ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 39.57% | 2 | 1 | ⚠️ |
| investigations_ordered | 0.00% | 0 | 1 | ❌ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 48.39% | 1 | 1 | ⚠️ |

**Chief Complaint:**
- Predicted: `breathless and it's getting worse`
- Ground Truth: `progressive breathlessness`

**Diagnosis:**
- Predicted: `congestive heart failure (suspected)`
- Ground Truth: `congestive heart failure (suspected)`

---

### Case: DER0001

**Summary Similarity:** 24.29%

**Predicted Summary:** A 45-year-old male with a history of diabetes, high cholesterol, and chronic back pain presents with a painful, swollen, and red rash on his right ankle. The patient reports poor glycemic control (A1C ~9) and describes systemic symptoms including polyuria, hunger, fatigue, and a feeling of being hot...

**Ground Truth Summary:** A 45-year-old male with poorly controlled diabetes presents with a painful, swollen, and red rash on his right ankle of one week's duration, now with purulent drainage. Patient also reports constitutional symptoms including fatigue, increased hunger, and a subjective feeling of fever over the last 1...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 71.05% | 1 | 1 | ✅ |
| symptoms | 20.06% | 10 | 11 | ❌ |
| symptom_onset | 23.53% | 1 | 1 | ❌ |
| medical_history | 67.49% | 5 | 5 | ⚠️ |
| current_medications | 44.03% | 4 | 3 | ⚠️ |
| allergies | 52.34% | 1 | 1 | ⚠️ |
| physical_exam_findings | 38.16% | 5 | 4 | ⚠️ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 22.73% | 3 | 2 | ❌ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 100.00% | 2 | 2 | ✅ |
| investigations_ordered | 100.00% | 0 | 0 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 57.47% | 1 | 1 | ⚠️ |

**Chief Complaint:**
- Predicted: `Painful rash on the right ankle`
- Ground Truth: `Painful, strange-looking rash on right ankle.`

---

### Case: GAS0001

**Summary Similarity:** 18.95%

**Predicted Summary:** The patient is a sexually active female presenting with a 9-day history of nausea, vomiting, and urinary frequency. She reports abdominal cramping and a 6-week delay in her menstrual cycle, though she reports she is not currently on hormonal birth control.

**Ground Truth Summary:** Patient is a female presenting with a 9-day history of persistent nausea and vomiting, accompanied by urinary frequency and abdominal cramping. Patient is sexually active with a male partner and reports amenorrhea (last period 6 weeks ago).

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 33.33% | 1 | 1 | ⚠️ |
| symptoms | 96.97% | 5 | 5 | ✅ |
| symptom_onset | 58.82% | 1 | 1 | ⚠️ |
| medical_history | 100.00% | 0 | 0 | ✅ |
| current_medications | 100.00% | 1 | 1 | ✅ |
| allergies | 100.00% | 1 | 1 | ✅ |
| physical_exam_findings | 100.00% | 0 | 0 | ✅ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 81.16% | 4 | 3 | ✅ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 68.80% | 2 | 1 | ⚠️ |
| investigations_ordered | 100.00% | 0 | 0 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 50.56% | 1 | 1 | ⚠️ |

**Chief Complaint:**
- Predicted: `Nausea`
- Ground Truth: `Persistent nausea and vomiting`

---

### Case: GAS0004

**Summary Similarity:** 34.73%

**Predicted Summary:** The patient is a student presenting with a 3-4 day history of frequent loose stools (diarrhea) occurring approximately hourly, accompanied by nausea, occasional vomiting, and recent-onset abdominal cramping. Patient denies blood in stool or vomit, urinary changes, or fever/chills, but reports genera...

**Ground Truth Summary:** The patient is a student presenting with a 3-4 day history of frequent, loose stools, abdominal cramping, and intermittent nausea/vomiting. There is no evidence of blood in stool or vomit, and the patient reports no recent travel or sick contacts, though they did dine at a new Chinese restaurant 5 d...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 100.00% | 1 | 1 | ✅ |
| symptoms | 59.38% | 7 | 7 | ⚠️ |
| symptom_onset | 81.48% | 1 | 1 | ✅ |
| medical_history | 100.00% | 1 | 1 | ✅ |
| current_medications | 47.27% | 1 | 1 | ⚠️ |
| allergies | 0.00% | 0 | 1 | ❌ |
| physical_exam_findings | 40.00% | 2 | 1 | ⚠️ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 64.81% | 2 | 2 | ⚠️ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 55.32% | 1 | 1 | ⚠️ |
| investigations_ordered | 100.00% | 0 | 0 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 100.00% | 0 | 0 | ✅ |

**Chief Complaint:**
- Predicted: `Diarrhea`
- Ground Truth: `Diarrhea`

---

### Case: GEN0001

**Summary Similarity:** 18.28%

**Predicted Summary:** A 30-year-old female presents with symptoms suggestive of a recurrent urinary tract infection, including dysuria and cloudy urine. She has a known history of overactive bladder treated with Botox injections and asthma. The clinical evaluation is ongoing, with further investigations and a physical ex...

**Ground Truth Summary:** 30-year-old female with a history of overactive bladder presents with symptoms suggestive of a urinary tract infection, specifically dysuria and increased frequency. She has a history of asthma and prior appendectomy, and manages her bladder condition with regular Botox injections. The doctor is con...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 100.00% | 1 | 1 | ✅ |
| symptoms | 81.01% | 6 | 6 | ✅ |
| symptom_onset | 91.43% | 1 | 1 | ✅ |
| medical_history | 100.00% | 3 | 3 | ✅ |
| current_medications | 93.44% | 2 | 2 | ✅ |
| allergies | 72.00% | 1 | 1 | ✅ |
| physical_exam_findings | 26.38% | 1 | 6 | ❌ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 55.71% | 4 | 2 | ⚠️ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 89.55% | 2 | 2 | ✅ |
| investigations_ordered | 55.56% | 1 | 1 | ⚠️ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 73.53% | 1 | 1 | ✅ |

**Chief Complaint:**
- Predicted: `Possible bladder infection`
- Ground Truth: `Possible bladder infection`

---

### Case: MSK0001

**Summary Similarity:** 23.53%

**Predicted Summary:** A retired male presents with acute lower back pain radiating to the right leg, triggered by lifting groceries. He reports stabbing pain worsened by movement or coughing and denies any neurologic deficits, fever, or trauma history. Medical history is significant for Type 2 diabetes, treated with insu...

**Ground Truth Summary:** The patient, a retired school teacher, presents with acute stabbing lower back pain radiating to the right leg, which began after lifting groceries. He reports the pain is aggravated by movement and coughing, and notes minimal relief from Tylenol. Physical examination is pending, and the differentia...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 100.00% | 1 | 1 | ✅ |
| symptoms | 61.80% | 3 | 3 | ⚠️ |
| symptom_onset | 40.91% | 1 | 1 | ⚠️ |
| medical_history | 73.42% | 2 | 2 | ✅ |
| current_medications | 100.00% | 2 | 2 | ✅ |
| allergies | 100.00% | 0 | 0 | ✅ |
| physical_exam_findings | 29.63% | 1 | 2 | ❌ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 0.00% | 0 | 2 | ❌ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 0.00% | 0 | 1 | ❌ |
| investigations_ordered | 100.00% | 0 | 0 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 100.00% | 0 | 0 | ✅ |

**Chief Complaint:**
- Predicted: `Low back pain`
- Ground Truth: `Low back pain`

---

### Case: MSK0025

**Summary Similarity:** 8.42%

**Predicted Summary:** The patient, a 42-year-old teacher, presented with severe lower back pain radiating to both legs and urinary incontinence following a motor vehicle accident two months ago. Clinical assessment suggests potential Cauda Equina Syndrome due to worsening pain, bilateral lower limb weakness, and saddle a...

**Ground Truth Summary:** The patient presents with worsening lower back pain following a motor vehicle accident two months ago, now accompanied by saddle anesthesia, urinary incontinence, and bilateral lower limb weakness. The doctor suspects cauda equina syndrome and has initiated an urgent diagnostic evaluation, including...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 72.88% | 1 | 1 | ✅ |
| symptoms | 64.94% | 7 | 8 | ⚠️ |
| symptom_onset | 61.45% | 1 | 1 | ⚠️ |
| medical_history | 26.17% | 1 | 2 | ❌ |
| current_medications | 35.29% | 1 | 3 | ⚠️ |
| allergies | 100.00% | 1 | 1 | ✅ |
| physical_exam_findings | 49.37% | 4 | 4 | ⚠️ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 1 | 1 | ✅ |
| differential_diagnosis | 49.50% | 4 | 4 | ⚠️ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 70.10% | 4 | 4 | ✅ |
| investigations_ordered | 89.47% | 1 | 1 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 40.46% | 1 | 1 | ⚠️ |

**Chief Complaint:**
- Predicted: `Severe lower back pain and urinary incontinence.`
- Ground Truth: `Worsening lower back pain, saddle anesthesia, and urinary incontinence`

**Diagnosis:**
- Predicted: `Suspected Cauda Equina Syndrome`
- Ground Truth: `Suspected Cauda Equina Syndrome`

---

### Case: RES0001

**Summary Similarity:** 18.44%

**Predicted Summary:** A 52-year-old female presents with an 8-day history of sharp chest pain and exertional shortness of breath, exacerbated by deep breathing. The patient reports a recent hysterectomy and sedentary recovery period, with ongoing symptoms including a dry cough, palpitations, and left leg soreness, raisin...

**Ground Truth Summary:** A 52-year-old female presents with 8 days of sharp chest pain exacerbated by deep breathing and progressive exertional dyspnea, recently having undergone a hysterectomy. The physical assessment and patient history raise concern for a possible pulmonary embolism given the recent surgery and reported ...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 100.00% | 1 | 1 | ✅ |
| symptoms | 57.66% | 6 | 7 | ⚠️ |
| symptom_onset | 100.00% | 1 | 1 | ✅ |
| medical_history | 55.56% | 2 | 2 | ⚠️ |
| current_medications | 38.46% | 2 | 2 | ⚠️ |
| allergies | 22.22% | 1 | 1 | ❌ |
| physical_exam_findings | 55.45% | 2 | 2 | ⚠️ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 0 | 0 | ✅ |
| differential_diagnosis | 91.97% | 4 | 4 | ✅ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 99.54% | 2 | 2 | ✅ |
| investigations_ordered | 100.00% | 0 | 0 | ✅ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 100.00% | 0 | 0 | ✅ |

**Chief Complaint:**
- Predicted: `Chest pain and trouble breathing`
- Ground Truth: `Chest pain and trouble breathing`

---

### Case: RES0088

**Summary Similarity:** 17.34%

**Predicted Summary:** The patient's child presents with a worsening hoarse voice and sore throat for the past 2-3 days, following a recent dry cough that has largely resolved. The doctor suspects viral laryngitis and plans to conduct a physical examination to rule out secondary bacterial infection, recommending vocal res...

**Ground Truth Summary:** The patient's daughter presents with a 2-3 day history of hoarse voice and sore throat, following a dry cough that began 4 days ago. The doctor suspects viral laryngitis and plans a physical examination of the oropharynx and neck to rule out bacterial infection, advising voice rest as part of the in...

| Field | Similarity | Pred Items | GT Items | Match? |
|-------|------------|------------|----------|--------|
| chief_complaint | 60.00% | 1 | 1 | ⚠️ |
| symptoms | 88.76% | 6 | 6 | ✅ |
| symptom_onset | 62.02% | 1 | 1 | ⚠️ |
| medical_history | 20.95% | 3 | 1 | ❌ |
| current_medications | 100.00% | 1 | 1 | ✅ |
| allergies | 80.00% | 1 | 1 | ✅ |
| physical_exam_findings | 38.17% | 2 | 3 | ⚠️ |
| vital_signs | 100.00% | 0 | 0 | ✅ |
| diagnosis | 100.00% | 1 | 1 | ✅ |
| differential_diagnosis | 87.50% | 2 | 2 | ✅ |
| prescribed_medications | 100.00% | 0 | 0 | ✅ |
| treatment_plan | 70.33% | 2 | 3 | ✅ |
| investigations_ordered | 0.00% | 1 | 0 | ❌ |
| referrals | 100.00% | 0 | 0 | ✅ |
| follow_up | 54.63% | 1 | 1 | ⚠️ |

**Chief Complaint:**
- Predicted: `Hoarse voice and sore throat`
- Ground Truth: `Hoarse voice`

**Diagnosis:**
- Predicted: `Laryngitis`
- Ground Truth: `Laryngitis`

---

## 5. Key Observations & Takeaways

### What Worked Well

1. **High-similarity fields:** 14 out of 15 SOAP fields achieved >50% average text similarity between the audio-based pipeline and the transcript-based ground truth.
2. **Encounter type consistency:** 223/223 (100%) cases had matching encounter types.
3. **Volume:** Successfully generated 271 predicted SOAP reports from audio files, demonstrating the pipeline scales across multiple specialties (CAR, DER, GAS, GEN, MSK, RES).
4. **Confidence:** The extraction model consistently reported high confidence for both predicted and ground truth SOAP notes.

### Known Limitations

1. **CUDA library issue at startup:** The first batch run failed for 12 files due to `libcublas.so.12` not being found. A re-run resolved this.
2. **UTF-16 encoded transcripts:** 2 ground truth transcript files (RES0002, RES0054) were encoded in UTF-16 and could not be read, preventing comparison for those cases.
3. **Incomplete run:** The batch stopped before processing all cases (likely due to Gemini API token quota exhaustion). 1 ground truth cases have no corresponding predicted SOAP report.
4. **Text similarity != semantic equivalence:** The SequenceMatcher-based similarity metric captures surface-level text overlap. Two SOAP notes may express the same clinical information using different phrasing, resulting in lower similarity scores than the actual semantic agreement.

### Recommendations for Future Work

1. **Semantic similarity metrics:** Consider using embedding-based similarity (e.g., sentence-transformers) for more meaningful comparison of clinical text.
2. **Fix UTF-16 handling:** Add a fallback encoding detection in the batch script to handle non-UTF-8 transcript files.
3. **CUDA environment validation:** Add a startup check in the batch script to verify GPU libraries are accessible before processing begins.
4. **Gemini token budgeting:** Track API token usage and implement batching/rate-limiting to avoid quota exhaustion mid-run.
