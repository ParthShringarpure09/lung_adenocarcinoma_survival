"""
cibersort.py — DIY CIBERSORT immune cell deconvolution.

Takes a patient's gene expression vector and the LM22 signature
matrix, and returns the fraction of 22 immune cell types present
in the tumour microenvironment.

This implementation took 3 rounds to get right in the original
project. The key steps are:
1. Back-transform log2 to linear space before deconvolution
2. Unit-normalise both the signature matrix and patient vector
3. Tune the nu parameter across [0.25, 0.5, 0.75]
4. Clip negative weights and fall back to NNLS if all zero
5. Normalise final weights to sum to 1.0
"""

import numpy as np
import pandas as pd
from sklearn.svm import NuSVR
from sklearn.preprocessing import normalize
from scipy.optimize import nnls


def load_lm22(lm22_path: str) -> pd.DataFrame:
    """
    Load the LM22 signature matrix from a tab-separated file.

    The LM22 matrix has genes as rows and 22 immune cell types
    as columns. Returns a DataFrame indexed by gene name.

    Args:
        lm22_path: Full path to LM22.txt file.

    Returns:
        DataFrame with shape (n_genes, 22), indexed by gene name.
    """
    lm22 = pd.read_csv(lm22_path, sep="\t", index_col=0)
    lm22.index = lm22.index.str.strip().str.upper()
    return lm22


def run_cibersort_single(
    patient_expr: pd.Series,
    lm22_matrix: pd.DataFrame
) -> dict:
    """
    Run CIBERSORT deconvolution for a single patient.

    Takes a patient gene expression Series (index = gene names,
    values = log2 expression) and the LM22 signature matrix,
    and returns a dictionary of 22 immune cell fractions that
    sum to 1.0.

    Args:
        patient_expr : pd.Series with gene names as index,
                       log2-normalised expression values.
        lm22_matrix  : pd.DataFrame from load_lm22(), shape
                       (n_genes, 22).

    Returns:
        Dictionary mapping immune cell type name to fraction
        (float between 0 and 1). All 22 values sum to 1.0.
    """
    # Find genes present in both patient data and LM22
    common_genes = patient_expr.index.intersection(lm22_matrix.index)

    if len(common_genes) < 10:
        # Not enough overlap — return uniform distribution as fallback
        n_cells = lm22_matrix.shape[1]
        return dict(zip(lm22_matrix.columns,
                        np.ones(n_cells) / n_cells))

    # Subset to common genes
    patient_sub = patient_expr.loc[common_genes]
    lm22_sub = lm22_matrix.loc[common_genes]

    # Step 1: Back-transform log2 to linear space
    # log2(x+1) → x, then clip to avoid negatives from float errors
    expr_linear = (2 ** patient_sub.values) - 1
    expr_linear = np.clip(expr_linear, 0, 1e6)

    lm22_linear = (2 ** lm22_sub.values) - 1
    lm22_linear = np.clip(lm22_linear, 0, 1e6)

    # Step 2: Unit-normalise (closed gene space normalisation)
    lm22_norm = normalize(lm22_linear, axis=0)
    expr_norm = normalize(expr_linear.reshape(1, -1))[0]

    # Step 3: Tune nu parameter across candidate values
    best_nu = 0.5
    best_error = np.inf

    for nu in [0.25, 0.5, 0.75]:
        try:
            svr = NuSVR(nu=nu, kernel="linear", C=1.0)
            svr.fit(lm22_norm, expr_norm)
            predicted = lm22_norm @ svr.coef_[0]
            error = np.mean((predicted - expr_norm) ** 2)
            if error < best_error:
                best_error = error
                best_nu = nu
        except Exception:
            continue

    # Step 4: Final SVR with best nu
    try:
        svr = NuSVR(nu=best_nu, kernel="linear", C=1.0)
        svr.fit(lm22_norm, expr_norm)
        raw_weights = svr.coef_[0]
    except Exception:
        raw_weights = np.zeros(lm22_matrix.shape[1])

    # Step 5: Clip negatives — NNLS fallback if all weights are zero
    clipped = np.maximum(raw_weights, 0)

    if clipped.sum() == 0:
        try:
            clipped, _ = nnls(lm22_norm, expr_norm)
            clipped = np.maximum(clipped, 0)
        except Exception:
            clipped = np.zeros(lm22_matrix.shape[1])

    # Step 6: Normalise to sum to 1.0
    total = clipped.sum()
    if total > 0:
        final = clipped / total
    else:
        # Last resort: uniform distribution
        final = np.ones(len(clipped)) / len(clipped)

    return dict(zip(lm22_matrix.columns, final))


def run_cibersort_batch(
    expression_df: pd.DataFrame,
    lm22_path: str
) -> pd.DataFrame:
    """
    Run CIBERSORT deconvolution for multiple patients at once.

    Useful for generating demo patient immune profiles in bulk.

    Args:
        expression_df : DataFrame with genes as rows, patients
                        as columns. Values are log2 expression.
        lm22_path     : Full path to LM22.txt file.

    Returns:
        DataFrame with patients as rows and 22 immune cell types
        as columns. Each row sums to 1.0.
    """
    lm22 = load_lm22(lm22_path)
    results = {}

    for patient_id in expression_df.columns:
        patient_expr = expression_df[patient_id].copy()
        patient_expr.index = patient_expr.index.str.strip().str.upper()
        fractions = run_cibersort_single(patient_expr, lm22)
        results[patient_id] = fractions

    result_df = pd.DataFrame(results).T
    return result_df


def get_immune_summary(fractions: dict) -> dict:
    """
    Extract key immune cell values for display in the app.

    Returns a summary with the most clinically relevant cells
    highlighted — M2 macrophages (immunosuppressive) and
    CD8 T cells (anti-tumour).

    Args:
        fractions: Dictionary from run_cibersort_single().

    Returns:
        Dictionary with keys:
        - 'M2_macrophages'   : float, fraction of M2 macrophages
        - 'CD8_T_cells'      : float, fraction of CD8 T cells
        - 'all_fractions'    : dict, full 22-cell profile
        - 'immunosuppressive': bool, True if M2 > 0.15
    """
    m2 = fractions.get("Macrophages M2", 0.0)
    cd8 = fractions.get("T cells CD8", 0.0)

    return {
        "M2_macrophages": m2,
        "CD8_T_cells": cd8,
        "all_fractions": fractions,
        "immunosuppressive": m2 > 0.15
    }