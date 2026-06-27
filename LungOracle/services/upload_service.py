def run_predict(file_obj, age, gender, stage):
    if file_obj is None:
        return EMPTY_STATE, None, None, None, ""
    try:
        try:
            df = pd.read_csv(file_obj, sep=None, engine="python",
                             index_col=0, header=None)
            expr_series = df.iloc[:, 0].astype(float)
        except Exception:
            df = pd.read_csv(file_obj, sep="\t", index_col=0, header=None)
            expr_series = df.iloc[:, 0].astype(float)
        expr_series.index = expr_series.index.astype(str).str.strip()
    except Exception as e:
        return (
            f"<div style='background:#fff5f5;border-left:4px solid #c53030;"
            f"border-radius:0 8px 8px 0;padding:20px 24px;"
            f"font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;'>"
            f"<div style='font-size:13px;font-weight:700;color:#c53030;"
            f"margin-bottom:6px;'>File parse error</div>"
            f"<div style='font-size:13px;color:#4a5568;'>{str(e)}</div></div>",
            None, None, None, ""
        )
    return run_prediction(
        expr_series=expr_series, age=age, gender=gender, stage=stage,
        survival_days=0, patient_label="Uploaded patient"
    )