"""
Reusable HTML components for LungOracle.

Every function returns an HTML string that can be embedded
inside Gradio HTML components.
"""


def section_header(kicker: str, title: str, description: str = "") -> str:
    """Large section heading."""

    desc = ""

    if description:
        desc = f"""
        <div class="lo-description">
            {description}
        </div>
        """

    return f"""
    <div class="lo-section-header">

        <div class="lo-kicker">
            {kicker}
        </div>

        <div class="lo-title">
            {title}
        </div>

        {desc}

    </div>
    """


def badge(text: str, icon: str = "✓") -> str:
    """Rounded badge."""

    return f"""
    <span class="lo-badge">
        <span>{icon}</span>
        <span>{text}</span>
    </span>
    """


def metric_card(value: str,
                label: str,
                accent: str = "#2563eb") -> str:
    """Metric/statistics card."""

    return f"""
    <div class="lo-card lo-metric-card">

        <div
            class="lo-metric-value"
            style="color:{accent};"
        >
            {value}
        </div>

        <div class="lo-metric-label">
            {label}
        </div>

    </div>
    """


def feature_card(icon: str,
                 title: str,
                 description: str) -> str:
    """Feature description card."""

    return f"""
    <div class="lo-card lo-feature-card">

        <div class="lo-feature-icon">
            {icon}
        </div>

        <div class="lo-feature-title">
            {title}
        </div>

        <div class="lo-feature-description">
            {description}
        </div>

    </div>
    """


def info_card(title: str,
              body: str,
              icon: str = "ℹ️") -> str:
    """General information card."""

    return f"""
    <div class="lo-card lo-info-card">

        <div class="lo-info-title">

            <span class="lo-info-icon">
                {icon}
            </span>

            {title}

        </div>

        <div class="lo-info-body">

            {body}

        </div>

    </div>
    """


def pipeline_step(icon: str,
                  title: str) -> str:
    """Research pipeline block."""

    return f"""
    <div class="lo-pipeline-step">

        <div class="lo-icon">
            {icon}
        </div>

        <div class="lo-pipeline-title">
            {title}
        </div>

    </div>
    """


def divider(label: str = "") -> str:
    """Section divider."""

    if label:

        return f"""
        <div class="lo-divider">

            <span>{label}</span>

        </div>
        """

    return """
    <hr class="lo-divider-line">
    """


def bullet(text: str,
           color: str = "#22c55e") -> str:
    """Small checklist item."""

    return f"""
    <div class="lo-bullet">

        <span
            class="lo-bullet-dot"
            style="background:{color};"
        ></span>

        <span>

            {text}

        </span>

    </div>
    """


def workflow_arrow() -> str:
    """Arrow used in research pipeline."""

    return """
    <div class="lo-workflow-arrow">

        ➜

    </div>
    """


def empty_placeholder(title: str,
                      subtitle: str) -> str:
    """Placeholder shown before prediction."""

    return f"""
    <div class="lo-empty">

        <div class="lo-empty-icon">

            🤖

        </div>

        <div class="lo-empty-title">

            {title}

        </div>

        <div class="lo-empty-subtitle">

            {subtitle}

        </div>

    </div>
    """