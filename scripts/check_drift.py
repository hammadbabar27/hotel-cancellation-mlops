"""Detect feature drift from recent inputs; nonzero exit signals alert, not automatic retrain."""
from pathlib import Path
import sys
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
log_file = ROOT / 'predictions.csv'
if not log_file.exists():
    print('No live requests yet; send >= 30 predictions first')
    sys.exit(0)
reference = pd.read_csv(ROOT / 'data' / 'hotel_reservations.csv')['lead_time'].dropna()
recent = pd.read_csv(log_file)['lead_time'].dropna().tail(100)
if len(recent) < 30:
    print(f'Not enough requests for drift check: {len(recent)}/30')
    sys.exit(0)
p = ks_2samp(reference, recent).pvalue
print(f'Lead-time data drift KS p-value: {p:.6f}')
if p < 0.05:
    print('DRIFT detected: investigate, collect labeled data and retrain after review')
    sys.exit(1)
print('No statistically significant lead-time drift detected')
