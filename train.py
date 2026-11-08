"""Train and compare three classifiers; save champion and measurable quality gate."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'hotel_reservations.csv'
ARTIFACTS = ROOT / 'artifacts'
TARGET = 'booking_status'
CATEGORICAL = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
NUMERIC = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
    'no_of_week_nights', 'required_car_parking_space', 'lead_time',
    'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest',
    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
    'avg_price_per_room', 'no_of_special_requests'
]
FEATURES = NUMERIC + CATEGORICAL


def preprocess():
    numerical = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', numerical, NUMERIC), ('cat', categorical, CATEGORICAL)])


def main():
    if not DATA.exists():
        raise SystemExit('Missing data. First run: python scripts/download_data.py')
    df = pd.read_csv(DATA)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Missing dataset columns: {sorted(missing)}')
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].map({'Canceled': 1, 'Not_Canceled': 0})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError('Expected two labels: Canceled, Not_Canceled')
    X = df[FEATURES]
    # Stratified fixed split ensures a fair and repeatable comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    candidates = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=42, class_weight='balanced_subsample'),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=130, random_state=42),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    # Local MLflow database. In GitHub CI it creates a fresh experiment per workflow run.
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', f'sqlite:///{(ROOT / "mlflow.db").as_posix()}'))
    mlflow.set_experiment('hotel-cancellation-prediction')
    best = None
    scores = {}
    for name, estimator in candidates.items():
        model = Pipeline([('preprocessor', preprocess()), ('classifier', estimator)])
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = {
                'accuracy': float(accuracy_score(y_test, predictions)),
                'precision': float(precision_score(y_test, predictions, zero_division=0)),
                'recall': float(recall_score(y_test, predictions, zero_division=0)),
                'f1': float(f1_score(y_test, predictions, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, probabilities)),
            }
            mlflow.log_params({'model_type': name, 'test_size': 0.2, 'random_state': 42,
                               'rows_in_dataset': len(df)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle"
)
            scores[name] = metrics
            print(f'{name}: F1={metrics["f1"]:.4f}, ROC-AUC={metrics["roc_auc"]:.4f}')
            if best is None or (metrics['f1'], metrics['roc_auc']) > (best[1]['f1'], best[1]['roc_auc']):
                best = (name, metrics, model)
    name, metrics, champion = best
    joblib.dump(champion, ARTIFACTS / 'champion.pkl')
    (ARTIFACTS / 'metrics.json').write_text(json.dumps({'champion': name, 'test_rows': len(X_test),
        'rows': len(df), 'metrics': metrics, 'all_models': scores}, indent=2))
    print(f'CHAMPION: {name} | F1={metrics["f1"]:.4f} | ROC-AUC={metrics["roc_auc"]:.4f}')

if __name__ == '__main__':
    main()
