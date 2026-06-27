import gradio as gr


def prediction_outputs(risk_html, km, shap, immune, explanation):
    visible = km is not None

    return (
        risk_html,
        gr.update(value=km, visible=visible),
        gr.update(value=shap, visible=visible),
        gr.update(value=immune, visible=visible),
        gr.update(value=explanation, visible=visible),
    )