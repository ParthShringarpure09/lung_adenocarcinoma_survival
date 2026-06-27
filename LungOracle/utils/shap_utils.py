"""
shap_utils.py — SHAP feature importance computation for LungOracle.

Computes SHAP values for a single patient prediction using the
GradientBoostingSurvivalAnalysis model. Returns the top features
driving the prediction with plain English labels for display.

Note: GradientBoostingSurvivalAnalysis requires shap.Explainer
with model.predict as the prediction function, not TreeExplainer.
"""

import numpy as np
import pandas as pd
import shap

# ── Plain English labels for all 124 features ─────────────────────────────
FEATURE_LABELS = {
    # Expression genes
    "EPGN_expr":       "EPGN Gene Expression",
    "SLC47A1_expr":    "SLC47A1 Gene Expression",
    "SIX1_expr":       "SIX1 Gene Expression",
    "RHCG_expr":       "RHCG Gene Expression",
    "GPC6_expr":       "GPC6 Gene Expression",
    "CCDC40_expr":     "CCDC40 Gene Expression",
    "CRHR2_expr":      "CRHR2 Gene Expression",
    "SH3TC2_expr":     "SH3TC2 Gene Expression",
    "HS3ST2_expr":     "HS3ST2 Gene Expression",
    "ACSM5_expr":      "ACSM5 Gene Expression",
    "CASP14_expr":     "CASP14 Gene Expression",
    "TSPAN11_expr":    "TSPAN11 Gene Expression",
    "CNTN3_expr":      "CNTN3 Gene Expression",
    "ALX1_expr":       "ALX1 Gene Expression",
    "TRPC2_expr":      "TRPC2 Gene Expression",
    "BEND5_expr":      "BEND5 Gene Expression",
    "STK33_expr":      "STK33 Gene Expression",
    "ERO1LB_expr":     "ERO1LB Gene Expression",
    "IRF4_expr":       "IRF4 Gene Expression",
    "NAV3_expr":       "NAV3 Gene Expression",
    "TG_expr":         "TG Gene Expression",
    "SERPINA6_expr":   "SERPINA6 Gene Expression",
    "CD70_expr":       "CD70 Gene Expression",
    "LST-3TM12_expr":  "LST-3TM12 Gene Expression",
    "MPV17L_expr":     "MPV17L Gene Expression",
    "LTK_expr":        "LTK Gene Expression",
    "NOS3_expr":       "NOS3 Gene Expression",
    "PDE10A_expr":     "PDE10A Gene Expression",
    "TMEM213_expr":    "TMEM213 Gene Expression",
    "NLRP2_expr":      "NLRP2 Gene Expression",
    "KCNA6_expr":      "KCNA6 Gene Expression",
    "PCSK9_expr":      "PCSK9 Gene Expression",
    "C8orf47_expr":    "C8orf47 Gene Expression",
    "SLCO1B1_expr":    "SLCO1B1 Gene Expression",
    "TMEM215_expr":    "TMEM215 Gene Expression",
    "ZNF257_expr":     "ZNF257 Gene Expression",
    "MT1A_expr":       "MT1A Gene Expression",
    "PAQR5_expr":      "PAQR5 Gene Expression",
    "GSTA3_expr":      "GSTA3 Gene Expression",
    "KNDC1_expr":      "KNDC1 Gene Expression",
    "WFDC3_expr":      "WFDC3 Gene Expression",
    "KIR2DL3_expr":    "KIR2DL3 Gene Expression",
    "C10orf90_expr":   "C10orf90 Gene Expression",
    "GAS2_expr":       "GAS2 Gene Expression",
    "HOXB9_expr":      "HOXB9 Gene Expression",
    "NCAM2_expr":      "NCAM2 Gene Expression",
    "HIST1H3G_expr":   "HIST1H3G Gene Expression",
    "ATP10B_expr":     "ATP10B Gene Expression",
    "HTR2B_expr":      "HTR2B Gene Expression",
    "SPRR1B_expr":     "SPRR1B Gene Expression",
    "NPAS2_expr":      "NPAS2 Gene Expression",
    "FGA_expr":        "FGA Gene Expression",
    "GPR64_expr":      "GPR64 Gene Expression",
    "SPRR2A_expr":     "SPRR2A Gene Expression",
    "TGFB2_expr":      "TGFB2 Gene Expression",
    "PKHD1L1_expr":    "PKHD1L1 Gene Expression",
    "S100A12_expr":    "S100A12 Gene Expression",
    "VCAN_expr":       "VCAN Gene Expression",
    "ABCA17P_expr":    "ABCA17P Gene Expression",
    "MBL1P_expr":      "MBL1P Gene Expression",
    "BLK_expr":        "BLK Gene Expression",
    "AUTS2_expr":      "AUTS2 Gene Expression",
    "CDH10_expr":      "CDH10 Gene Expression",
    "C1orf141_expr":   "C1orf141 Gene Expression",
    "TMEM139_expr":    "TMEM139 Gene Expression",
    "ODZ1_expr":       "ODZ1 Gene Expression",
    "IGFBPL1_expr":    "IGFBPL1 Gene Expression",
    "GSTT2_expr":      "GSTT2 Gene Expression",
    "ADAMTS1_expr":    "ADAMTS1 Gene Expression",
    "CYP2D7P1_expr":   "CYP2D7P1 Gene Expression",
    "FBN3_expr":       "FBN3 Gene Expression",
    "LOC654433_expr":  "LOC654433 Gene Expression",
    # Dysregulation features
    "LINGO2_dysreg":   "LINGO2 Tumour Dysregulation",
    "CD109_dysreg":    "CD109 Tumour Dysregulation",
    "EPGN_dysreg":     "EPGN Tumour Dysregulation",
    "RGS20_dysreg":    "RGS20 Tumour Dysregulation",
    "COL4A6_dysreg":   "COL4A6 Tumour Dysregulation",
    "UNC5D_dysreg":    "UNC5D Tumour Dysregulation",
    "DKK1_dysreg":     "DKK1 Tumour Dysregulation",
    "GADD45G_dysreg":  "GADD45G Tumour Dysregulation",
    "IGFBP1_dysreg":   "IGFBP1 Tumour Dysregulation",
    "GABRA2_dysreg":   "GABRA2 Tumour Dysregulation",
    "KLHDC7A_dysreg":  "KLHDC7A Tumour Dysregulation",
    "SIX1_dysreg":     "SIX1 Tumour Dysregulation",
    "GPC6_dysreg":     "GPC6 Tumour Dysregulation",
    "IRX5_dysreg":     "IRX5 Tumour Dysregulation",
    "KRT6C_dysreg":    "KRT6C Tumour Dysregulation",
    "ELF5_dysreg":     "ELF5 Tumour Dysregulation",
    "CLDN3_dysreg":    "CLDN3 Tumour Dysregulation",
    "SH3TC2_dysreg":   "SH3TC2 Tumour Dysregulation",
    "CXCL17_dysreg":   "CXCL17 Tumour Dysregulation",
    "SDK1_dysreg":     "SDK1 Tumour Dysregulation",
    # Immune cell types
    "B cells naive":                    "Naive B Cells",
    "B cells memory":                   "Memory B Cells",
    "Plasma cells":                     "Plasma Cells",
    "T cells CD8":                      "CD8 T Cells (Anti-Tumour)",
    "T cells CD4 naive":                "Naive CD4 T Cells",
    "T cells CD4 memory resting":       "Resting Memory CD4 T Cells",
    "T cells CD4 memory activated":     "Activated Memory CD4 T Cells",
    "T cells follicular helper":        "Follicular Helper T Cells",
    "T cells regulatory (Tregs)":       "Regulatory T Cells (Tregs)",
    "T cells gamma delta":              "Gamma Delta T Cells",
    "NK cells resting":                 "Resting NK Cells",
    "NK cells activated":               "Activated NK Cells",
    "Monocytes":                        "Monocytes",
    "Macrophages M0":                   "M0 Macrophages (Unactivated)",
    "Macrophages M1":                   "M1 Macrophages (Inflammatory)",
    "Macrophages M2":                   "M2 Macrophages (Immunosuppressive)",
    "Dendritic cells resting":          "Resting Dendritic Cells",
    "Dendritic cells activated":        "Activated Dendritic Cells",
    "Mast cells resting":               "Resting Mast Cells",
    "Mast cells activated":             "Activated Mast Cells",
    "Eosinophils":                      "Eosinophils",
    "Neutrophils":                      "Neutrophils",
    # Clinical features
    "age":              "Patient Age",
    "gender":           "Patient Sex",
    "stage_Stage II":   "Cancer Stage II",
    "stage_Stage III":  "Cancer Stage III",
    "stage_Stage IV":   "Cancer Stage IV",
    # Interaction features
    "stageIII_x_M2":   "Stage III + High M2 Interaction",
    "stageIV_x_CD8":   "Stage IV + CD8 T Cell Interaction",
    "age_x_stageIII":  "Age × Stage III Interaction",
    "stageIII_x_Treg": "Stage III + Treg Interaction",
    "M2_x_CD8":        "M2 Macrophage × CD8 T Cell Ratio",
}


def compute_shap_values(
    feature_vector: pd.Series,
    xgb_model,
    scaler,
    n_background: int = 100
) -> tuple[np.ndarray, float]:
    """
    Compute SHAP values for a single patient using the survival model.

    Uses shap.Explainer with model.predict as the prediction function.
    This is required for GradientBoostingSurvivalAnalysis — TreeExplainer
    does not support this model type.

    A background dataset of random normal samples is used as the
    reference distribution for SHAP computation.

    Args:
        feature_vector : pd.Series with exactly 124 features.
        xgb_model      : Loaded GradientBoostingSurvivalAnalysis model.
        scaler         : Loaded StandardScaler.
        n_background   : Number of background samples for SHAP
                         (default 100, higher = slower but more accurate).

    Returns:
        Tuple of (shap_values array of shape (124,), base_value float).
    """
    X = feature_vector.values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Build background dataset from random normal samples
    np.random.seed(42)
    background = np.random.normal(0, 1, (n_background, X_scaled.shape[1]))

    # Use shap.Explainer with predict function
    explainer = shap.Explainer(xgb_model.predict, background)
    shap_values = explainer(X_scaled)

    values = shap_values.values[0]
    base_value = float(shap_values.base_values[0])

    return values, base_value


def get_top_shap_features(
    feature_vector: pd.Series,
    shap_values: np.ndarray,
    n_top: int = 10
) -> pd.DataFrame:
    """
    Return the top N features by absolute SHAP value with plain English labels.

    Positive SHAP values increase the predicted risk.
    Negative SHAP values decrease the predicted risk.

    Args:
        feature_vector : pd.Series with 124 features (pre-scaling).
        shap_values    : np.ndarray of shape (124,) from compute_shap_values.
        n_top          : Number of top features to return (default 10).

    Returns:
        DataFrame with columns:
        - 'feature'      : raw feature name
        - 'label'        : plain English label
        - 'shap_value'   : SHAP value (positive = increases risk)
        - 'feature_value': actual value of the feature for this patient
        - 'abs_shap'     : absolute SHAP value
        - 'direction'    : 'Increases Risk' or 'Decreases Risk'
        Sorted by abs_shap descending.
    """
    feature_names = list(feature_vector.index)

    df = pd.DataFrame({
        "feature":       feature_names,
        "shap_value":    shap_values,
        "feature_value": feature_vector.values,
    })

    df["abs_shap"]  = df["shap_value"].abs()
    df["label"]     = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df["direction"] = df["shap_value"].apply(
        lambda v: "Increases Risk" if v > 0 else "Decreases Risk"
    )

    df = df.sort_values("abs_shap", ascending=False).head(n_top)
    df = df.reset_index(drop=True)

    return df


def get_shap_explanation_text(top_features_df: pd.DataFrame) -> str:
    """
    Generate a one-sentence plain English explanation of the SHAP chart.

    Identifies the single most important risk-increasing and
    risk-decreasing feature and names them in the explanation.

    Args:
        top_features_df: DataFrame from get_top_shap_features().

    Returns:
        A plain English string suitable for display below the chart.
    """
    risk_up   = top_features_df[top_features_df["direction"] == "Increases Risk"]
    risk_down = top_features_df[top_features_df["direction"] == "Decreases Risk"]

    if len(risk_up) > 0 and len(risk_down) > 0:
        top_up   = risk_up.iloc[0]["label"]
        top_down = risk_down.iloc[0]["label"]
        return (f"The strongest risk-increasing factor for this patient was "
                f"{top_up}, while {top_down} was the strongest protective factor.")
    elif len(risk_up) > 0:
        top_up = risk_up.iloc[0]["label"]
        return f"All top features increased this patient's predicted risk, led by {top_up}."
    else:
        top_down = risk_down.iloc[0]["label"]
        return f"All top features decreased this patient's predicted risk, led by {top_down}."