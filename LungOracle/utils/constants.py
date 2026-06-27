from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

MODEL_DIR = ROOT / "models"

DEMO_CSV = DATA_DIR / "demo_patients.csv"

ALL_RISK_JSON = DATA_DIR / "all_risk_scores.json"

BASE_CLINICAL = DATA_DIR / "base_clinical.csv"

BASE_EXPR = DATA_DIR / "base_expression.csv"

MODEL_PATH = MODEL_DIR / "model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"