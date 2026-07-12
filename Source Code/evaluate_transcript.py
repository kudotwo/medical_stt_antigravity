"""
evaluate_transcript.py — Cross-check pipeline output against clean reference transcript

Metrics computed:
  1. WER  (Word Error Rate)         — transcription accuracy
  2. CER  (Character Error Rate)    — fine-grained text accuracy
  3. Speaker accuracy               — how well Doctor/Patient labels match D/P
  4. Per-line diff table            — side-by-side reference vs output

Usage:
    python evaluate_transcript.py

Outputs:
    - Printed report to console
    - evaluation_report.txt  (detailed per-line diff)
"""

import re
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE, '..', 'audio_sample', 'audio_recordings')

REFERENCE_PATH = os.path.join(AUDIO_DIR, 'Clean_Transcripts', 'CAR0001.txt')
OUTPUT_PATH    = os.path.join(AUDIO_DIR, 'Audio_Recordings', 'CAR0001_transcript.txt')
REPORT_PATH    = os.path.join(BASE, 'evaluation_report.txt')


# ═══════════════════════════════════════════════════════════════════════════════
# Parsers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_reference(path: str) -> list[dict]:
    """
    Parse clean reference transcript.
    Format: 'D: text'  or  'P: text'  (blank lines between)
    Returns: [{"speaker": "Doctor"|"Patient", "text": "..."}]
    """
    entries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('D:'):
                entries.append({'speaker': 'Doctor', 'text': line[2:].strip()})
            elif line.startswith('P:'):
                entries.append({'speaker': 'Patient', 'text': line[2:].strip()})
    return entries


def parse_output(path: str) -> list[dict]:
    """
    Parse pipeline output transcript.
    Format: '[0.0s->8.9s] Doctor: text'
    Returns: [{"speaker": "Doctor"|"Patient", "text": "..."}]
    """
    entries = []
    pattern = re.compile(r'^\[[\d.]+s->[\d.]+s\]\s+(Doctor|Patient):\s*(.*)')
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                entries.append({'speaker': m.group(1), 'text': m.group(2).strip()})
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# WER / CER helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein edit distance between two lists."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = edit_distance(words) / len(reference_words)."""
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    if not ref_words:
        return 0.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(reference_chars)."""
    ref_chars = list(_normalize(reference))
    hyp_chars = list(_normalize(hypothesis))
    if not ref_chars:
        return 0.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ═══════════════════════════════════════════════════════════════════════════════
# Speaker accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def compute_speaker_accuracy(ref_entries: list[dict], out_entries: list[dict]) -> dict:
    """
    Compare speaker labels line-by-line (up to the shorter list).
    Returns dict with accuracy stats.
    """
    pairs = list(zip(ref_entries, out_entries))
    correct = sum(1 for r, o in pairs if r['speaker'] == o['speaker'])
    total   = len(pairs)
    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total else 0.0,
        'mismatches': [(i, r['speaker'], o['speaker']) for i, (r, o) in enumerate(pairs) if r['speaker'] != o['speaker']]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate():
    print('=' * 70)
    print('  Transcript Evaluation Report')
    print('=' * 70)

    # Load
    ref = parse_reference(REFERENCE_PATH)
    out = parse_output(OUTPUT_PATH)

    print(f'\nReference entries : {len(ref)}')
    print(f'Pipeline entries  : {len(out)}')

    # ── Full-text WER / CER ──────────────────────────────────────────────────
    ref_full = ' '.join(e['text'] for e in ref)
    out_full = ' '.join(e['text'] for e in out)

    wer = compute_wer(ref_full, out_full)
    cer = compute_cer(ref_full, out_full)

    print(f'\n{"─"*70}')
    print('TRANSCRIPTION ACCURACY (full text)')
    print(f'{"─"*70}')
    print(f'  WER (Word Error Rate)      : {wer*100:.1f}%   (lower = better, 0% = perfect)')
    print(f'  CER (Character Error Rate) : {cer*100:.1f}%')
    print(f'  Word accuracy              : {(1-wer)*100:.1f}%')

    # ── Speaker diarization accuracy ─────────────────────────────────────────
    spk = compute_speaker_accuracy(ref, out)
    print(f'\n{"─"*70}')
    print('SPEAKER DIARIZATION ACCURACY')
    print(f'{"─"*70}')
    print(f'  Correct labels : {spk["correct"]} / {spk["total"]}')
    print(f'  Accuracy       : {spk["accuracy"]*100:.1f}%')
    if spk['mismatches']:
        print(f'  Mismatches     : {len(spk["mismatches"])} lines')
        for idx, ref_spk, out_spk in spk['mismatches'][:10]:
            ref_txt = ref[idx]["text"][:60] + '...' if len(ref[idx]["text"]) > 60 else ref[idx]["text"]
            print(f'    Line {idx+1:3d}: expected {ref_spk:<8} got {out_spk:<8}  "{ref_txt}"')
    else:
        print('  No mismatches! Perfect speaker labeling.')

    # ── Per-speaker WER ──────────────────────────────────────────────────────
    print(f'\n{"─"*70}')
    print('PER-SPEAKER TRANSCRIPTION ACCURACY')
    print(f'{"─"*70}')
    for spk_name in ['Doctor', 'Patient']:
        ref_text = ' '.join(e['text'] for e in ref if e['speaker'] == spk_name)
        out_text = ' '.join(e['text'] for e in out if e['speaker'] == spk_name)
        w = compute_wer(ref_text, out_text)
        print(f'  {spk_name:<10}: WER {w*100:.1f}%  ({(1-w)*100:.1f}% accuracy)')

    # ── Side-by-side diff (first 20 pairs) ──────────────────────────────────
    print(f'\n{"─"*70}')
    print('SIDE-BY-SIDE COMPARISON (first 20 lines)')
    print(f'{"─"*70}')
    print(f'{"#":<4}  {"REFERENCE":<45}  {"PIPELINE OUTPUT":<45}  {"SPK?":<6}  {"WER":<6}')
    print(f'{"─"*4}  {"─"*45}  {"─"*45}  {"─"*6}  {"─"*6}')

    report_lines = []
    for i, (r, o) in enumerate(zip(ref, out)):
        spk_match = '✓' if r['speaker'] == o['speaker'] else '✗'
        line_wer  = compute_wer(r['text'], o['text'])
        ref_disp  = f'[{r["speaker"][0]}] {r["text"]}'[:45]
        out_disp  = f'[{o["speaker"][0]}] {o["text"]}'[:45]
        row = f'{i+1:<4}  {ref_disp:<45}  {out_disp:<45}  {spk_match:<6}  {line_wer*100:.0f}%'
        report_lines.append(row)
        if i < 20:
            print(row)

    if len(ref) > 20:
        print(f'  ... ({len(ref)-20} more lines in evaluation_report.txt)')

    # ── Save full report ─────────────────────────────────────────────────────
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('TRANSCRIPT EVALUATION REPORT\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Reference : {REFERENCE_PATH}\n')
        f.write(f'Output    : {OUTPUT_PATH}\n\n')
        f.write(f'WER  : {wer*100:.1f}%\n')
        f.write(f'CER  : {cer*100:.1f}%\n')
        f.write(f'Speaker Accuracy: {spk["accuracy"]*100:.1f}%\n\n')
        f.write(f'{"#":<4}  {"REF SPEAKER":<12}  {"OUT SPEAKER":<12}  {"SPK?":<5}  {"WER":>5}  {"REFERENCE TEXT":<50}  OUTPUT TEXT\n')
        f.write('─' * 120 + '\n')
        for i, (r, o) in enumerate(zip(ref, out)):
            spk_match = 'OK' if r['speaker'] == o['speaker'] else 'FAIL'
            line_wer  = compute_wer(r['text'], o['text'])
            f.write(f'{i+1:<4}  {r["speaker"]:<12}  {o["speaker"]:<12}  {spk_match:<5}  {line_wer*100:>4.0f}%  {r["text"]:<50}  {o["text"]}\n')

    print(f'\n  Full report saved: {REPORT_PATH}')
    print('\n' + '=' * 70)


if __name__ == '__main__':
    evaluate()
