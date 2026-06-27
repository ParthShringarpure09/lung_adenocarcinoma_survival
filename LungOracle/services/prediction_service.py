


def run_prediction(expr_series, age, gender, stage, survival_days, patient_label):
    result = run_inference(
        expr_input=expr_series,
        age=float(age), gender=gender, stage=stage, models=MODELS
    )

    if result["overlap_error"]:
        err = f"""
        <div style="background:#fff5f5;border-left:4px solid #c53030;
                    border-radius:0 8px 8px 0;padding:20px 24px;
                    font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;">
          <div style="font-size:13px;font-weight:700;color:#c53030;margin-bottom:6px;">
            Insufficient gene data
          </div>
          <div style="font-size:13px;color:#4a5568;">{result['overlap_message']}</div>
        </div>"""
        return err, None, None, None, ""

    risk_score  = result["risk_score"]
    risk_group  = result["risk_group"]
    immune_frac = result["immune_fractions"]

    col_map = {
        "HIGH":   ("#c53030", "#fed7d7", "#fc8181"),
        "MEDIUM": ("#c05621", "#feebc8", "#f6ad55"),
        "LOW":    ("#276749", "#c6f6d5", "#68d391"),
    }
    text_col, bg_col, border_col = col_map[risk_group]

    warn_parts = []
    if result["overlap_message"].startswith("WARNING"):
        warn_parts.append(result["overlap_message"])
    if result["stage_warning"]:
        warn_parts.append(result["stage_warning"])
    warn_str = " | ".join(warn_parts)

    m2_val  = immune_frac.get("Macrophages M2", 0)
    cd8_val = immune_frac.get("T cells CD8", 0)
    immuno  = "Immunosuppressive microenvironment" if m2_val > 0.15 else ""

    risk_html = f"""
    <div style="background:{bg_col};border:1.5px solid {border_col};
                border-left:5px solid {text_col};border-radius:8px;
                padding:24px 28px;
                font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;">
      <div style="display:flex;align-items:flex-start;gap:28px;flex-wrap:wrap;">
        <div style="min-width:120px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
                      text-transform:uppercase;color:{text_col};margin-bottom:4px;">
            Predicted risk
          </div>
          <div style="font-size:56px;font-weight:800;color:{text_col};
                      line-height:1;margin-bottom:6px;letter-spacing:-1px;">
            {risk_group}
          </div>
          <div style="font-size:22px;font-weight:700;color:{text_col};">
            {risk_score:.3f}
          </div>
        </div>
        <div style="flex:1;min-width:200px;padding-top:2px;">
          <div style="font-size:13px;color:#4a5568;line-height:2.2;margin-bottom:12px;">
            <span style="color:#718096;">Age</span>&nbsp;
            <span style="font-weight:600;color:#2d3748;">{int(age)} years</span>
            &nbsp;&nbsp;
            <span style="color:#718096;">Sex</span>&nbsp;
            <span style="font-weight:600;color:#2d3748;">{gender}</span>
            &nbsp;&nbsp;
            <span style="color:#718096;">Stage</span>&nbsp;
            <span style="font-weight:600;color:#2d3748;">{stage}</span>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <span style="background:#fff5f5;color:#c53030;font-size:12px;
                         font-weight:600;padding:5px 12px;border-radius:4px;
                         border:1px solid #fc8181;">
              M2 macrophages &nbsp;{m2_val:.1%}
            </span>
            <span style="background:#f0fff4;color:#276749;font-size:12px;
                         font-weight:600;padding:5px 12px;border-radius:4px;
                         border:1px solid #68d391;">
              CD8 T cells &nbsp;{cd8_val:.1%}
            </span>
            {"<span style='font-size:12px;color:#c53030;font-style:italic;'>" + immuno + "</span>" if immuno else ""}
          </div>
        </div>
      </div>
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid {border_col};
                  font-size:11px;color:#718096;">
        Thresholds: &nbsp;
        <span style="color:#276749;font-weight:700;">LOW</span> &lt;0.4
        &nbsp;|&nbsp;
        <span style="color:#c05621;font-weight:700;">MEDIUM</span> 0.4–0.6
        &nbsp;|&nbsp;
        <span style="color:#c53030;font-weight:700;">HIGH</span> &gt;0.6
        {"&nbsp;&nbsp;<span style='color:#c05621;'>&#9888; " + warn_str + "</span>" if warn_str else ""}
      </div>
    </div>"""

    km_fig = plot_survival_curve(
        KM_HIGH_DF, KM_LOW_DF,
        patient_risk_score=risk_score,
        patient_survival_days=survival_days if survival_days else 0,
        patient_risk_group=risk_group,
        patient_label=patient_label
    )

    shap_vals, base_val = compute_shap_values(
        result["feature_vector"], MODELS["xgb"], MODELS["scaler"],
        n_background=100
    )
    top_features = get_top_shap_features(result["feature_vector"], shap_vals, n_top=10)
    explanation  = get_shap_explanation_text(top_features)
    shap_fig     = plot_shap_waterfall(top_features)
    immune_fig   = plot_immune_profile(immune_frac, patient_label=patient_label)

    return risk_html, km_fig, shap_fig, immune_fig, explanation
