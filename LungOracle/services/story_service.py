def update_story(short_name):
    if not short_name:
        return ""
    row = DEMO_DF[DEMO_DF["short_name"] == short_name].iloc[0]
    risk_col = {
        "HIGH": "#c53030", "MEDIUM": "#c05621", "LOW": "#276749"
    }.get(str(row["risk_group"]), "#718096")
    event_text = "Deceased" if row["event"] == 1 else "Alive / censored"
    surv_years = round(row["survival_days"] / 365.25, 1)
    return f"""
    <div style="background:#ffffff;border:1.5px solid #e2e8f0;border-radius:8px;
                padding:14px 16px;margin-top:8px;
                font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;">
      <div style="font-size:15px;font-weight:700;color:#1a365d;margin-bottom:4px;">
        {short_name.split(' — ')[0]}
      </div>
      <div style="font-size:12px;color:#718096;margin-bottom:8px;">
        {row['stage']} &nbsp;·&nbsp; {int(row['age'])} years &nbsp;·&nbsp; {row['gender']}
      </div>
      <div style="font-size:12px;color:#4a5568;line-height:1.5;margin-bottom:10px;">
        {row['label']}
      </div>
      <div style="display:flex;gap:16px;padding-top:8px;border-top:1px solid #e2e8f0;">
        <div style="font-size:11px;color:#718096;">
          Survival&nbsp;<span style="font-weight:700;color:#2d3748;">{surv_years}y</span>
        </div>
        <div style="font-size:11px;color:#718096;">
          Outcome&nbsp;<span style="font-weight:700;color:#2d3748;">{event_text}</span>
        </div>
        <div style="font-size:11px;color:#718096;">
          Risk&nbsp;<span style="font-weight:700;color:{risk_col};">{row['risk_group']}</span>
        </div>
      </div>
    </div>"""
