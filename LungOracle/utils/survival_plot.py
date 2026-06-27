"""
survival_plot.py — Publication-grade plotting for LungOracle.
"""

import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
import pandas as pd

# Clinical Publication Palette
COLOR_HIGH = "#8B0000"  # Maroon / Brick red
COLOR_LOW  = "#475569"  # Slate grey
COLOR_MARK = "#0F172A"  # Near Black

# Basic font styling for a clean academic look
FONT_FAMILY = "Arial, sans-serif"

def build_km_data(clinical_path, risk_scores):
    """Build KM curves for high and low risk TCGA groups."""
    clin = pd.read_csv(clinical_path)
    clin = clin.rename(columns={"patient.bcr_patient_barcode": "patient_id"})
    clin["patient_id"] = clin["patient_id"].str.upper()
    clin["risk_score"] = clin["patient_id"].map(risk_scores)
    clin = clin.dropna(subset=["risk_score"])

    median_score = clin["risk_score"].median()
    high = clin[clin["risk_score"] >= median_score].copy()
    low  = clin[clin["risk_score"] <  median_score].copy()

    high["survival_years"] = high["survival_time"] / 365.25
    low["survival_years"]  = low["survival_time"]  / 365.25

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(durations=high["survival_years"], event_observed=high["event"])
    kmf_low.fit(durations=low["survival_years"], event_observed=low["event"])

    high_df = kmf_high.survival_function_.reset_index()
    high_df.columns = ["timeline", "KM_estimate"]
    low_df  = kmf_low.survival_function_.reset_index()
    low_df.columns  = ["timeline", "KM_estimate"]

    return high_df, low_df

def get_patient_km_position(risk_score, survival_days, high_df, low_df):
    """Helper to find where the patient star should be placed on the KM curve."""
    x_years = survival_days / 365.25
    km_df   = high_df if risk_score >= 0.5 else low_df
    before  = km_df[km_df["timeline"] <= x_years]
    y_prob  = float(before["KM_estimate"].iloc[-1]) if len(before) > 0 else 1.0
    return x_years, y_prob

def plot_survival_curve(high_df, low_df, patient_risk_score, patient_survival_days, patient_risk_group, patient_label="This Patient"):
    fig = go.Figure()

    # High Risk Baseline - Plain line, no fill
    fig.add_trace(go.Scatter(
        x=high_df["timeline"], y=high_df["KM_estimate"],
        mode="lines", name="High risk cohort",
        line=dict(color=COLOR_HIGH, width=2)
    ))

    # Low Risk Baseline
    fig.add_trace(go.Scatter(
        x=low_df["timeline"], y=low_df["KM_estimate"],
        mode="lines", name="Low risk cohort",
        line=dict(color=COLOR_LOW, width=2, dash='dash')
    ))

    # Patient Marker
    if patient_survival_days and patient_survival_days > 0:
        star_x, star_y = get_patient_km_position(patient_risk_score, patient_survival_days, high_df, low_df)
    else:
        star_x, star_y = 0.0, 1.0

    fig.add_trace(go.Scatter(
        x=[star_x], y=[star_y], mode="markers", name="Patient Profile",
        marker=dict(symbol="circle", size=10, color=COLOR_MARK)
    ))

    # Clean, academic layout
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            title="Time (Years)", 
            showgrid=False, 
            linecolor="black", 
            linewidth=1,
            title_font=dict(family=FONT_FAMILY, size=12, color="black"),
            tickfont=dict(family=FONT_FAMILY, size=11, color="black")
        ),
        yaxis=dict(
            title="Survival Probability", 
            showgrid=False, 
            linecolor="black", 
            linewidth=1, 
            range=[0, 1.05], 
            tickformat=".1f",
            title_font=dict(family=FONT_FAMILY, size=12, color="black"),
            tickfont=dict(family=FONT_FAMILY, size=11, color="black")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family=FONT_FAMILY, size=11, color="black")
        ),
        height=400
    )
    return fig

def plot_shap_waterfall(top_features_df):
    """Academic style horizontal bar chart."""
    fig = go.Figure()
    
    colors = [COLOR_HIGH if val > 0 else COLOR_LOW for val in top_features_df['shap_value']]
    
    fig.add_trace(go.Bar(
        x=top_features_df['shap_value'],
        y=top_features_df['label'],
        orientation='h',
        marker_color=colors,
    ))
    
    fig.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="SHAP Value (Impact)", showgrid=False, linecolor="black", zerolinecolor="black"),
        yaxis=dict(title="", showgrid=False, linecolor="black", autorange="reversed"),
        height=350, showlegend=False
    )
    return fig

def plot_immune_profile(immune_fractions, patient_label="Patient"):
    """Clean dot plot."""
    sorted_cells = sorted(immune_fractions.items(), key=lambda x: x[1])
    cells = [c[0] for c in sorted_cells[-10:]] # Show only top 10 for clinical clarity
    values = [c[1] for c in sorted_cells[-10:]]
    
    fig = go.Figure()
    
    for cell, val in zip(cells, values):
        fig.add_shape(type="line", x0=0, x1=val, y0=cell, y1=cell, line=dict(color="#E2E8F0", width=1))
        
    fig.add_trace(go.Scatter(
        x=values, y=cells, mode="markers",
        marker=dict(color=COLOR_MARK, size=8)
    ))
    
    fig.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=20, t=10, b=0),
        xaxis=dict(title="Cell Fraction", showgrid=False, linecolor="black", tickformat=".2f"),
        yaxis=dict(title="", showgrid=False, linecolor="black"),
        height=350, showlegend=False
    )
    return fig