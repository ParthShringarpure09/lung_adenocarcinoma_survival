"""
inference.py — Full 9-step inference pipeline for LungOracle.

Takes a patient's gene expression data and clinical information,
runs through all preprocessing steps, and returns a risk score
with all intermediate outputs needed for visualisation.

The 124 features are assembled in this exact order:
  0-71  : 72 Lasso expression genes       (_expr suffix)
  72-91 : 20 dysregulation z-scores       (_dysreg suffix)
  92-113: 22 immune cell fractions        (LM22 column names)
  114-118: 5 clinical features            (age, gender, stages)
  119-123: 5 interaction features         (cross terms)
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"

XGBOOST_PATH     = f"{MODELS_DIR}/xgboost_final.pkl"
SCALER_PATH      = f"{MODELS_DIR}/scaler_xgb.pkl"
COX_PATH         = f"{MODELS_DIR}/cox_lasso_expression.pkl"
GENE_LIST_PATH   = f"{MODELS_DIR}/gene_list.json"
DYSREG_PATH      = f"{MODELS_DIR}/top_dysreg_genes.json"
LM22_PATH        = f"{DATA_DIR}/LM22.txt"
GTEX_PATH        = f"{DATA_DIR}/gtex_reference.json"

# ── Exact 72 Lasso genes in training order ─────────────────────────────────
LASSO_GENES = [
    'EPGN', 'SLC47A1', 'SIX1', 'RHCG', 'GPC6', 'CCDC40', 'CRHR2',
    'SH3TC2', 'HS3ST2', 'ACSM5', 'CASP14', 'TSPAN11', 'CNTN3', 'ALX1',
    'TRPC2', 'BEND5', 'STK33', 'ERO1LB', 'IRF4', 'NAV3', 'TG',
    'SERPINA6', 'CD70', 'LST-3TM12', 'MPV17L', 'LTK', 'NOS3', 'PDE10A',
    'TMEM213', 'NLRP2', 'KCNA6', 'PCSK9', 'C8orf47', 'SLCO1B1',
    'TMEM215', 'ZNF257', 'MT1A', 'PAQR5', 'GSTA3', 'KNDC1', 'WFDC3',
    'KIR2DL3', 'C10orf90', 'GAS2', 'HOXB9', 'NCAM2', 'HIST1H3G',
    'ATP10B', 'HTR2B', 'SPRR1B', 'NPAS2', 'FGA', 'GPR64', 'SPRR2A',
    'TGFB2', 'PKHD1L1', 'S100A12', 'VCAN', 'ABCA17P', 'MBL1P', 'BLK',
    'AUTS2', 'CDH10', 'C1orf141', 'TMEM139', 'ODZ1', 'IGFBPL1',
    'GSTT2', 'ADAMTS1', 'CYP2D7P1', 'FBN3', 'LOC654433'
]

# ── Exact 20 dysregulation genes in training order ─────────────────────────
DYSREG_GENES = [
    'LINGO2', 'CD109', 'EPGN', 'RGS20', 'COL4A6', 'UNC5D', 'DKK1',
    'GADD45G', 'IGFBP1', 'GABRA2', 'KLHDC7A', 'SIX1', 'GPC6', 'IRX5',
    'KRT6C', 'ELF5', 'CLDN3', 'SH3TC2', 'CXCL17', 'SDK1'
]

# ── Exact 22 immune cell types in training order ───────────────────────────
IMMUNE_CELLS = [
    'B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
    'T cells CD4 naive', 'T cells CD4 memory resting',
    'T cells CD4 memory activated', 'T cells follicular helper',
    'T cells regulatory (Tregs)', 'T cells gamma delta',
    'NK cells resting', 'NK cells activated', 'Monocytes',
    'Macrophages M0', 'Macrophages M1', 'Macrophages M2',
    'Dendritic cells resting', 'Dendritic cells activated',
    'Mast cells resting', 'Mast cells activated',
    'Eosinophils', 'Neutrophils'
]


def load_models():
    """
    Load all model artifacts from disk and return them as a dict.

    Loads the XGBoost survival model, the feature scaler, and
    the GTEx reference once at startup so they are reused across
    predictions without reloading from disk each time.

    Returns:
        Dictionary with keys: 'xgb', 'scaler', 'gtex', 'lm22'
    """
    with open(XGBOOST_PATH, "rb") as f:
        xgb = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(GTEX_PATH, "r") as f:
        gtex = json.load(f)

    from utils.cibersort import load_lm22
    lm22 = load_lm22(LM22_PATH)

    return {"xgb": xgb, "scaler": scaler, "gtex": gtex, "lm22": lm22}


# ── Step 1: Parse gene names ───────────────────────────────────────────────

def parse_gene_names(expr_series: pd.Series) -> pd.Series:
    """
    Clean and standardise gene names from a patient expression file.

    Handles common formats:
    - Entrez ID suffixes: EGFR|1956 → EGFR
    - Lowercase: egfr → EGFR
    - Whitespace padding: ' EGFR ' → 'EGFR'

    Args:
        expr_series: pd.Series with raw gene names as index.

    Returns:
        pd.Series with cleaned uppercase gene names as index.
    """
    new_index = []
    for gene in expr_series.index:
        gene = str(gene).strip()
        if "|" in gene:
            gene = gene.split("|")[0]
        gene = gene.upper()
        new_index.append(gene)

    expr_series = expr_series.copy()
    expr_series.index = new_index

    # Drop duplicates — keep first occurrence
    expr_series = expr_series[~expr_series.index.duplicated(keep="first")]

    return expr_series


# ── Step 2: Detect and apply normalisation ─────────────────────────────────

def detect_and_normalise(expr_series: pd.Series) -> tuple[pd.Series, str]:
    """
    Detect whether expression values are raw counts or log2-transformed.

    If the maximum value is greater than 25, the data is assumed to be
    raw counts and log2(x+1) transformation is applied. If max <= 25,
    the data is assumed to already be log2-transformed and is used as-is.

    Args:
        expr_series: pd.Series with numeric expression values.

    Returns:
        Tuple of (normalised_series, detection_message).
    """
    max_val = expr_series.max()

    if max_val > 25:
        normalised = np.log2(expr_series.clip(lower=0) + 1)
        message = (f"Raw counts detected (max={max_val:.1f}). "
                   f"Applied log2(x+1) transformation.")
    else:
        normalised = expr_series.copy()
        message = (f"Log2-transformed data detected (max={max_val:.1f}). "
                   f"Used as-is.")

    return normalised, message


# ── Step 3: Check gene overlap ─────────────────────────────────────────────

def check_gene_overlap(expr_series: pd.Series) -> tuple[int, str, bool]:
    """
    Check how many of the 72 required Lasso genes are in the patient file.

    Returns a warning if fewer than 80% are found (< 58 genes), and
    an error flag if fewer than 50% are found (< 36 genes).

    Args:
        expr_series: pd.Series with cleaned gene names as index.

    Returns:
        Tuple of (n_found, message, is_error).
        is_error is True if overlap is critically low (< 36 genes).
    """
    patient_genes = set(expr_series.index)
    required_genes = set(LASSO_GENES)
    found = patient_genes.intersection(required_genes)
    n_found = len(found)
    n_required = len(LASSO_GENES)

    if n_found < 36:
        message = (f"ERROR: Only {n_found} of {n_required} required genes "
                   f"found. Insufficient gene overlap for reliable prediction."
                   f" Please check your file format.")
        return n_found, message, True

    elif n_found < 58:
        message = (f"WARNING: Only {n_found} of {n_required} required genes "
                   f"found. Results may be less reliable.")
        return n_found, message, False

    else:
        message = (f"Gene overlap OK: {n_found} of {n_required} "
                   f"required genes found.")
        return n_found, message, False


# ── Step 4: Extract expression features ───────────────────────────────────

def extract_expression_features(expr_series: pd.Series) -> pd.Series:
    """
    Extract the 72 Lasso expression genes in exact training order.

    Genes present in the patient file are extracted directly.
    Genes missing from the patient file are filled with 0.0.
    The returned Series has column names with '_expr' suffix,
    matching exactly what the model was trained on.

    Args:
        expr_series: pd.Series with cleaned, normalised gene expression.

    Returns:
        pd.Series with exactly 72 values indexed as GENE_expr.
    """
    features = {}
    for gene in LASSO_GENES:
        feature_name = f"{gene}_expr"
        if gene in expr_series.index:
            features[feature_name] = float(expr_series[gene])
        else:
            features[feature_name] = 0.0

    return pd.Series(features)


# ── Step 5: Compute dysregulation features ────────────────────────────────

def extract_dysregulation_features(
    expr_series: pd.Series,
    gtex_reference: dict
) -> pd.Series:
    """
    Compute the 20 dysregulation z-score features in exact training order.

    For each consensus dysregulation gene, computes:
        z = (patient_expression - gtex_mean) / gtex_std

    Missing genes are filled with 0.0 (no dysregulation assumed).
    The returned Series has column names with '_dysreg' suffix.

    Args:
        expr_series    : pd.Series with cleaned, normalised expression.
        gtex_reference : Dict from gtex_reference.json with mean/std
                         per gene.

    Returns:
        pd.Series with exactly 20 values indexed as GENE_dysreg.
    """
    features = {}

    for gene in DYSREG_GENES:
        feature_name = f"{gene}_dysreg"

        if gene not in expr_series.index or gene not in gtex_reference:
            features[feature_name] = 0.0
            continue

        expr_val = float(expr_series[gene])
        ref = gtex_reference[gene]
        mean = ref.get("mean", 0.0)
        std  = ref.get("std", 1.0)

        if std < 1e-6:
            std = 1e-6

        features[feature_name] = (expr_val - mean) / std

    return pd.Series(features)


# ── Step 6: Run CIBERSORT ─────────────────────────────────────────────────

def extract_immune_features(
    expr_series: pd.Series,
    lm22_matrix: pd.DataFrame
) -> pd.Series:
    """
    Run DIY CIBERSORT to get 22 immune cell fractions in training order.

    Calls the CIBERSORT implementation from utils/cibersort.py and
    returns the 22 fractions in the exact column order the model
    was trained on.

    Args:
        expr_series : pd.Series with cleaned, normalised expression.
        lm22_matrix : pd.DataFrame loaded from LM22.txt.

    Returns:
        pd.Series with exactly 22 values, one per immune cell type.
    """
    from utils.cibersort import run_cibersort_single

    fractions = run_cibersort_single(expr_series, lm22_matrix)

    # Return in exact training order
    features = {cell: fractions.get(cell, 0.0) for cell in IMMUNE_CELLS}
    return pd.Series(features)


# ── Step 7: Build clinical features ───────────────────────────────────────

def build_clinical_features(
    age: float,
    gender: str,
    stage: str
) -> tuple[pd.Series, str]:
    """
    Build the 5 clinical features used by the model.

    Encodes gender as binary (1=Male, 0=Female) and stage as
    three dummy variables (Stage II, III, IV). If stage is
    Unknown, all stage dummies are set to 0 and a warning
    is returned.

    Args:
        age    : Patient age in years (float).
        gender : 'Male' or 'Female'.
        stage  : One of 'Stage I', 'Stage II', 'Stage III',
                 'Stage IV', 'Unknown'.

    Returns:
        Tuple of (pd.Series with 5 features, warning_message).
        warning_message is empty string if stage is known.
    """
    gender_val = 1.0 if gender == "Male" else 0.0

    stage_II  = 1.0 if stage == "Stage II"  else 0.0
    stage_III = 1.0 if stage == "Stage III" else 0.0
    stage_IV  = 1.0 if stage == "Stage IV"  else 0.0

    warning = ""
    if stage == "Unknown":
        warning = ("Stage is unknown. All stage features set to 0. "
                   "Prediction accuracy may be reduced.")

    features = pd.Series({
        "age":            float(age),
        "gender":         gender_val,
        "stage_Stage II": stage_II,
        "stage_Stage III": stage_III,
        "stage_Stage IV": stage_IV,
    })

    return features, warning


# ── Step 8: Compute interaction features ──────────────────────────────────

def build_interaction_features(
    immune_features: pd.Series,
    clinical_features: pd.Series
) -> pd.Series:
    """
    Compute the 5 interaction features used by the model.

    These capture joint effects between stage, immune cells,
    and age that are not captured by the individual features.

    Interactions:
        stageIII_x_M2   = Stage III dummy × M2 macrophage fraction
        stageIV_x_CD8   = Stage IV dummy × CD8 T cell fraction
        age_x_stageIII  = age × Stage III dummy
        stageIII_x_Treg = Stage III dummy × Treg fraction
        M2_x_CD8        = M2 macrophage fraction × CD8 T cell fraction

    Args:
        immune_features  : pd.Series from extract_immune_features().
        clinical_features: pd.Series from build_clinical_features().

    Returns:
        pd.Series with exactly 5 interaction feature values.
    """
    stage_III = clinical_features["stage_Stage III"]
    stage_IV  = clinical_features["stage_Stage IV"]
    age       = clinical_features["age"]

    m2   = immune_features.get("Macrophages M2", 0.0)
    cd8  = immune_features.get("T cells CD8", 0.0)
    treg = immune_features.get("T cells regulatory (Tregs)", 0.0)

    return pd.Series({
        "stageIII_x_M2":   stage_III * m2,
        "stageIV_x_CD8":   stage_IV  * cd8,
        "age_x_stageIII":  age       * stage_III,
        "stageIII_x_Treg": stage_III * treg,
        "M2_x_CD8":        m2        * cd8,
    })


# ── Step 9: Predict ────────────────────────────────────────────────────────

def predict_risk(
    feature_vector: pd.Series,
    xgb_model,
    scaler
) -> tuple[float, str]:
    """
    Scale features and run the XGBoost survival model to get a risk score.

    The risk score is the model's predicted risk (higher = worse prognosis).
    Risk groups: > 0.6 = HIGH, 0.4-0.6 = MEDIUM, < 0.4 = LOW.

    Args:
        feature_vector : pd.Series with exactly 124 features in
                         training order.
        xgb_model      : Loaded GradientBoostingSurvivalAnalysis model.
        scaler         : Loaded StandardScaler.

    Returns:
        Tuple of (risk_score float, risk_group string).
    """
    X = feature_vector.values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    raw_score = xgb_model.predict(X_scaled)[0]

    # Normalise to 0-1 range using sigmoid-like scaling
    # The model outputs a log-hazard ratio; we convert to a 0-1 risk score
    risk_score = float(1 / (1 + np.exp(-raw_score)))

    if risk_score > 0.6:
        risk_group = "HIGH"
    elif risk_score >= 0.4:
        risk_group = "MEDIUM"
    else:
        risk_group = "LOW"

    return risk_score, risk_group


# ── Full pipeline ──────────────────────────────────────────────────────────

def run_inference(
    expr_input,
    age: float,
    gender: str,
    stage: str,
    models: dict
) -> dict:
    """
    Run the complete 9-step inference pipeline for a single patient.

    This is the main entry point called by app.py. It accepts either
    a file path (string) or a pre-loaded pd.Series as expr_input.

    Args:
        expr_input : Either a file path string (CSV or TXT) or a
                     pd.Series with gene names as index and log2
                     expression values.
        age        : Patient age in years.
        gender     : 'Male' or 'Female'.
        stage      : 'Stage I', 'Stage II', 'Stage III',
                     'Stage IV', or 'Unknown'.
        models     : Dictionary from load_models().

    Returns:
        Dictionary with keys:
        - 'risk_score'        : float 0-1
        - 'risk_group'        : 'HIGH', 'MEDIUM', or 'LOW'
        - 'feature_vector'    : pd.Series of 124 features (pre-scale)
        - 'immune_fractions'  : dict of 22 cell type fractions
        - 'norm_message'      : str, normalisation detection message
        - 'overlap_message'   : str, gene overlap message
        - 'stage_warning'     : str, empty if stage is known
        - 'overlap_error'     : bool, True if gene overlap too low
        - 'age'               : float
        - 'gender'            : str
        - 'stage'             : str
    """
    xgb    = models["xgb"]
    scaler = models["scaler"]
    gtex   = models["gtex"]
    lm22   = models["lm22"]

    # ── Load expression data if a file path was provided ──────────────────
    if isinstance(expr_input, str):
        try:
            df = pd.read_csv(expr_input, sep=None, engine="python",
                             index_col=0, header=None)
            expr_series = df.iloc[:, 0].astype(float)
        except Exception as e:
            raise ValueError(f"Could not read expression file: {e}")
    else:
        expr_series = expr_input.copy()

    # Step 1: Parse gene names
    expr_series = parse_gene_names(expr_series)

    # Step 2: Detect normalisation
    expr_series, norm_message = detect_and_normalise(expr_series)

    # Step 3: Check gene overlap
    n_found, overlap_message, overlap_error = check_gene_overlap(expr_series)

    if overlap_error:
        return {
            "risk_score": None,
            "risk_group": None,
            "feature_vector": None,
            "immune_fractions": None,
            "norm_message": norm_message,
            "overlap_message": overlap_message,
            "stage_warning": "",
            "overlap_error": True,
            "age": age,
            "gender": gender,
            "stage": stage,
        }

    # Step 4: Extract expression features (72)
    expr_features = extract_expression_features(expr_series)

    # Step 5: Compute dysregulation features (20)
    dysreg_features = extract_dysregulation_features(expr_series, gtex)

    # Step 6: Run CIBERSORT (22 immune fractions)
    immune_features = extract_immune_features(expr_series, lm22)
    immune_fractions = immune_features.to_dict()

    # Step 7: Build clinical features (5)
    clinical_features, stage_warning = build_clinical_features(
        age, gender, stage
    )

    # Step 8: Compute interaction features (5)
    interaction_features = build_interaction_features(
        immune_features, clinical_features
    )

    # Step 9: Assemble all 124 features in exact training order
    feature_vector = pd.concat([
        expr_features,        # 72 features
        dysreg_features,      # 20 features
        immune_features,      # 22 features
        clinical_features,    # 5 features
        interaction_features  # 5 features
    ])

    # Verify feature count
    assert len(feature_vector) == 124, (
        f"Expected 124 features, got {len(feature_vector)}"
    )

    # Predict
    risk_score, risk_group = predict_risk(feature_vector, xgb, scaler)

    return {
        "risk_score":       risk_score,
        "risk_group":       risk_group,
        "feature_vector":   feature_vector,
        "immune_fractions": immune_fractions,
        "norm_message":     norm_message,
        "overlap_message":  overlap_message,
        "stage_warning":    stage_warning,
        "overlap_error":    False,
        "age":              age,
        "gender":           gender,
        "stage":            stage,
    }