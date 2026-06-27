def run_demo(short_name):
    if not short_name:
        return EMPTY_STATE, None, None, None, ""
    row = DEMO_DF[DEMO_DF["short_name"] == short_name].iloc[0]
    patient_id    = row["patient_id"]
    age           = row["age"]
    gender        = row["gender"]
    stage         = row["stage"]
    survival_days = row["survival_days"]
    if patient_id not in EXPR_MATRIX.index:
        return (
            f"<div style='color:#c53030;padding:20px;font-family:Inter,Arial,sans-serif;'>"
            f"Expression data not found for {patient_id}</div>",
            None, None, None, ""
        )
    expr_series = EXPR_MATRIX.loc[patient_id]
    short_label = short_name.split(" — ")[0]
    return run_prediction(expr_series, age, gender, stage, survival_days, short_label)
