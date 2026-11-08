import json
import joblib
from pathlib import Path
from train import FEATURES, DATA

ROOT = Path(__file__).resolve().parents[1]

def test_original_data_available():
    assert DATA.is_file(), 'Download dataset with python scripts/download_data.py'

def test_champion_model_saved():
    model = joblib.load(ROOT / 'artifacts' / 'champion.pkl')
    assert hasattr(model, 'predict_proba')

def test_quality_metrics():
    metrics = json.loads((ROOT / 'artifacts' / 'metrics.json').read_text())
    assert len(FEATURES) == 17
    assert 0 <= metrics['metrics']['f1'] <= 1
    assert len(metrics['all_models']) == 3
