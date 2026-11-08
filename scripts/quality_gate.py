"""CI fails on model underperformance, not merely failed unit tests."""
import json
from pathlib import Path
p = Path(__file__).resolve().parents[1] / 'artifacts' / 'metrics.json'
m = json.loads(p.read_text())['metrics']
MIN_F1 = 0.65
MIN_AUC = 0.75
print(f'Quality gate: F1 {m["f1"]:.3f} >= {MIN_F1}; AUC {m["roc_auc"]:.3f} >= {MIN_AUC}')
if m['f1'] < MIN_F1 or m['roc_auc'] < MIN_AUC:
    raise SystemExit('FAIL: quality gate did not pass')
print('PASS: model meets the minimum quality gate')
