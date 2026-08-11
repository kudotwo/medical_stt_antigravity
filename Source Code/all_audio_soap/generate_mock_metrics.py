import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
METRICS_CSV = BASE_DIR / 'model_comparison_metrics.csv'

CASES = ['CAR0001', 'MSK0005', 'RES0050', 'GAS0003', 'DER0001']

data = []

# Mock data showing gemini-3.6-flash generally outperforming 3.1-flash-lite
scores = {
    'gemini-3.1-flash-lite': {
        'CAR0001': [7, 7, 7],
        'MSK0005': [8, 7, 7],
        'RES0050': [7, 8, 8],
        'GAS0003': [7, 6, 6],
        'DER0001': [8, 8, 7],
    },
    'gemini-3.6-flash': {
        'CAR0001': [9, 9, 8],
        'MSK0005': [10, 9, 9],
        'RES0050': [9, 9, 9],
        'GAS0003': [9, 8, 8],
        'DER0001': [10, 10, 9],
    }
}

for model, case_scores in scores.items():
    for case, s in case_scores.items():
        data.append({
            'case_id': case,
            'model': model,
            'clinical_fidelity': s[0],
            'logical_consistency': s[1],
            'predictive_accuracy': s[2]
        })

with open(METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['case_id', 'model', 'clinical_fidelity', 'logical_consistency', 'predictive_accuracy'])
    writer.writeheader()
    writer.writerows(data)

print(f"Mock metrics saved to {METRICS_CSV}")
