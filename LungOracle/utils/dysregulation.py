"""
dysregulation.py — Tumour dysregulation z-score computation.

Measures how far each gene's expression in a patient deviates
from healthy tissue (GTEx reference). A high absolute z-score
means the gene is strongly dysregulated in this tumour compared
to normal lung tissue.

In training, the top 20 dysregulated genes were selected by
Cox p-value inside each cross-validation fold. At inference
time we use the consensus top 20 from training, stored in
models/top_dysreg_genes.json.
"""

import json
import numpy as np
import pandas as pd


def load_gtex_reference(gtex_path: str) -> dict:
    """
    Load the GTEx healthy tissue reference statistics.

    The reference file contains mean and standard deviation of
    gene expression from healthy lung tissue (GTEx database).
    These are used as the baseline for computing dysregulation.

    Args:
        gtex_path: Full path to gtex_reference.json file.

    Returns:
        Dictionary with gene names as keys. Each value is a
        dict with 'mean' and 'std' fields (both floats).
    """
    with open(gtex_path, "r") as f:
        reference = json.load(f)
    return reference


def load_top_dysreg_genes(genes_path: str) -> list:
    """
    Load the consensus top 20 dysregulation genes from training.

    During training, the top 20 genes were selected by Cox
    p-value inside each cross-validation fold. This function
    loads the consensus list so inference uses the same genes.

    Args:
        genes_path: Full path to top_dysreg_genes.json file.

    Returns:
        List of 20 gene name strings, in the exact order they
        appear in the training feature matrix.
    """
    with open(genes_path, "r") as f:
        genes = json.load(f)
    return genes


def compute_zscore(
    patient_expr: pd.Series,
    gtex_reference: dict
) -> pd.Series:
    """
    Compute dysregulation z-scores for all genes in a patient.

    For each gene present in both the patient data and the GTEx
    reference, computes:
        z = (patient_expression - gtex_mean) / gtex_std

    A positive z-score means the gene is overexpressed compared
    to healthy tissue. A negative z-score means underexpression.
    High absolute z-score means strong dysregulation.

    Args:
        patient_expr   : pd.Series with gene names as index
                         and log2 expression values.
        gtex_reference : Dictionary from load_gtex_reference().

    Returns:
        pd.Series with gene names as index and z-scores as
        values, for all genes present in both sources.
    """
    zscores = {}

    for gene, expr_val in patient_expr.items():
        gene_upper = str(gene).strip().upper()

        if gene_upper not in gtex_reference:
            continue

        ref = gtex_reference[gene_upper]
        mean = ref.get("mean", 0.0)
        std = ref.get("std", 1.0)

        # Avoid division by zero for genes with no variance
        if std < 1e-6:
            std = 1e-6

        z = (expr_val - mean) / std
        zscores[gene_upper] = z

    return pd.Series(zscores)


def extract_dysregulation_features(
    patient_expr: pd.Series,
    gtex_reference: dict,
    top_dysreg_genes: list
) -> pd.Series:
    """
    Extract the 20 dysregulation features used by the model.

    Computes z-scores for all available genes, then selects
    exactly the consensus top 20 genes from training. Missing
    genes are filled with 0.0 (no dysregulation assumed).

    The returned Series has column names in the format:
        GENENAME_dysreg
    matching exactly what the model was trained on.

    Args:
        patient_expr      : pd.Series, log2 expression values.
        gtex_reference    : Dictionary from load_gtex_reference().
        top_dysreg_genes  : List of 20 gene names from training.

    Returns:
        pd.Series with exactly 20 values, indexed as
        ['GENE1_dysreg', 'GENE2_dysreg', ..., 'GENE20_dysreg'].
    """
    # Normalise patient gene names to uppercase
    patient_expr = patient_expr.copy()
    patient_expr.index = patient_expr.index.str.strip().str.upper()

    # Compute z-scores for all available genes
    all_zscores = compute_zscore(patient_expr, gtex_reference)

    # Select the consensus top 20 genes in exact training order
    dysreg_features = {}

    for gene in top_dysreg_genes:
        gene_upper = str(gene).strip().upper()
        feature_name = f"{gene_upper}_dysreg"

        if gene_upper in all_zscores.index:
            dysreg_features[feature_name] = all_zscores[gene_upper]
        else:
            # Gene not in patient file — fill with 0 (no dysregulation)
            dysreg_features[feature_name] = 0.0

    return pd.Series(dysreg_features)


def get_top_dysregulated(
    patient_expr: pd.Series,
    gtex_reference: dict,
    n_top: int = 5
) -> pd.DataFrame:
    """
    Find the most dysregulated genes in this patient for display.

    Used in the app to show the user which genes are most
    abnormal compared to healthy lung tissue. Sorted by
    absolute z-score descending.

    Args:
        patient_expr   : pd.Series, log2 expression values.
        gtex_reference : Dictionary from load_gtex_reference().
        n_top          : Number of top genes to return (default 5).

    Returns:
        DataFrame with columns:
        - 'gene'       : gene name
        - 'zscore'     : dysregulation z-score
        - 'direction'  : 'Overexpressed' or 'Underexpressed'
        - 'abs_zscore' : absolute value of z-score
        Sorted by abs_zscore descending.
    """
    patient_expr = patient_expr.copy()
    patient_expr.index = patient_expr.index.str.strip().str.upper()

    all_zscores = compute_zscore(patient_expr, gtex_reference)

    df = pd.DataFrame({
        "gene": all_zscores.index,
        "zscore": all_zscores.values
    })

    df["abs_zscore"] = df["zscore"].abs()
    df["direction"] = df["zscore"].apply(
        lambda z: "Overexpressed" if z > 0 else "Underexpressed"
    )

    df = df.sort_values("abs_zscore", ascending=False).head(n_top)
    df = df.reset_index(drop=True)

    return df