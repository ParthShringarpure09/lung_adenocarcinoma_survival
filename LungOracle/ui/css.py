"""
Global CSS for LungOracle.

All reusable styling lives here.
"""


def get_css() -> str:
    return r"""

/* ==========================================================
   LUNGORACLE DESIGN SYSTEM
========================================================== */

:root{

    --primary:#2563eb;
    --primary-dark:#1d4ed8;

    --success:#16a34a;
    --warning:#d97706;
    --danger:#dc2626;

    --background:#f8fafc;
    --surface:#ffffff;

    --text:#0f172a;
    --muted:#64748b;

    --border:#e2e8f0;

    --radius-sm:12px;
    --radius:18px;
    --radius-lg:24px;

    --shadow-sm:
        0 2px 8px rgba(15,23,42,.04);

    --shadow:
        0 10px 28px rgba(15,23,42,.06);

    --shadow-lg:
        0 22px 60px rgba(15,23,42,.10);

}



/* ==========================================================
   BODY
========================================================== */

body{

    background:var(--background);

    color:var(--text);

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;

}



/* ==========================================================
   PAGE WIDTH
========================================================== */

.lo-container{

    max-width:1450px;

    margin:auto;

}



.lo-section{

    max-width:1450px;

    margin:auto;

    padding:70px 40px;

}



/* ==========================================================
   TYPOGRAPHY
========================================================== */

.lo-kicker{

    font-size:12px;

    font-weight:700;

    text-transform:uppercase;

    letter-spacing:1.5px;

    color:var(--primary);

    margin-bottom:12px;

}



.lo-title{

    font-size:42px;

    font-weight:800;

    letter-spacing:-1.5px;

    color:var(--text);

    margin-bottom:18px;

}



.lo-description{

    max-width:900px;

    font-size:18px;

    line-height:1.9;

    color:var(--muted);

}



/* ==========================================================
   CARD SYSTEM
========================================================== */

.lo-card{

    background:var(--surface);

    border:1px solid var(--border);

    border-radius:var(--radius);

    box-shadow:var(--shadow);

    transition:
        transform .25s,
        box-shadow .25s;

}



.lo-card:hover{

    transform:translateY(-5px);

    box-shadow:var(--shadow-lg);

}



/* ==========================================================
   FEATURE CARDS
========================================================== */

.lo-feature-card{

    padding:28px;

}



.lo-feature-icon{

    width:68px;

    height:68px;

    border-radius:16px;

    background:#eff6ff;

    display:flex;

    justify-content:center;

    align-items:center;

    font-size:34px;

}



.lo-feature-title{

    margin-top:22px;

    font-size:21px;

    font-weight:700;

}



.lo-feature-description{

    margin-top:12px;

    color:var(--muted);

    line-height:1.8;

}



/* ==========================================================
   METRIC CARD
========================================================== */

.lo-metric-card{

    padding:24px;

    text-align:center;

}



.lo-metric-value{

    font-size:54px;

    font-weight:600;

    letter-spacing:-2px;

    color:#12294f !important;

    line-height:1;

}



.lo-metric-label{

    margin-top:10px;

    font-size:14px;

    font-weight:500;

    color:#64748b !important;

}

.lo-metric-card{

    padding:34px 24px;

    text-align:center;

    min-height:150px;

    display:flex;

    flex-direction:column;

    justify-content:center;

}


/* ==========================================================
   BADGES
========================================================== */

.lo-badge{

    display:inline-flex;

    align-items:center;

    gap:8px;

    padding:10px 18px;

    background:#eff6ff;

    color:var(--primary);

    border-radius:999px;

    font-size:13px;

    font-weight:600;

}



/* ==========================================================
   GRID
========================================================== */

.lo-grid-2{

    display:grid;

    grid-template-columns:repeat(2,1fr);

    gap:24px;

}



.lo-grid-3{

    display:grid;

    grid-template-columns:repeat(3,1fr);

    gap:24px;

}



.lo-grid-4{

    display:grid;

    grid-template-columns:repeat(4,1fr);

    gap:24px;

}



/* ==========================================================
   PIPELINE
========================================================== */

.lo-pipeline{

    display:flex;

    justify-content:space-between;

    align-items:center;

    flex-wrap:wrap;

    gap:18px;

}



.lo-pipeline-step{

    flex:1;

    min-width:140px;

    text-align:center;

}



.lo-icon{

    width:72px;

    height:72px;

    margin:auto;

    border-radius:18px;

    background:#eff6ff;

    display:flex;

    justify-content:center;

    align-items:center;

    font-size:34px;

}



.lo-pipeline-title{

    margin-top:14px;

    font-weight:700;

}



.lo-workflow-arrow{

    font-size:28px;

    color:#94a3b8;

}



/* ==========================================================
   SIDEBAR
========================================================== */

.workspace-card{

    background:white;

    border:1px solid var(--border);

    border-radius:24px;

    padding:24px;

    box-shadow:var(--shadow);

    position:sticky;

    top:24px;

}

/* Sidebar statistics */

.lo-stat-card{
    background:#ffffff;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:16px 10px;
    text-align:center;
}

.lo-stat-value{
    font-size:38px;
    font-weight:800;
    color:#2563eb;
    line-height:1;
}

.lo-stat-label{
    margin-top:8px;
    font-size:13px;
    color:#64748b;
}

.lo-badge{
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    background:#eff6ff;
    color:#2563eb;
    font-size:11px;
    font-weight:600;
}


/* ==========================================================
   EMPTY STATE
========================================================== */

.lo-empty{

    text-align:center;

    padding:70px 40px;

}



.lo-empty-icon{

    font-size:70px;

    margin-bottom:22px;

}



.lo-empty-title{

    font-size:28px;

    font-weight:700;

    color:var(--text);

}



.lo-empty-subtitle{

    margin:18px auto 0;

    max-width:520px;

    color:var(--muted);

    line-height:1.8;

}



/* ==========================================================
   BUTTONS
========================================================== */

#analyse-btn,
#predict-analyse-btn{

    border-radius:14px !important;

    height:52px !important;

    font-weight:700 !important;

    font-size:15px !important;

}



/* ==========================================================
   PLOTS
========================================================== */

.lo-plot{

    border-radius:18px;

    overflow:hidden;

    box-shadow:var(--shadow);

}



/* ==========================================================
   DIVIDER
========================================================== */

.lo-divider-line{

    border:none;

    border-top:1px solid var(--border);

    margin:32px 0;

}



/* ==========================================================
   RESPONSIVE
========================================================== */

@media (max-width:1200px){

.lo-grid-4{

grid-template-columns:repeat(2,1fr);

}

}



@media (max-width:768px){

.lo-grid-4,
.lo-grid-3,
.lo-grid-2{

grid-template-columns:1fr;

}

.lo-title{

font-size:32px;

}

.lo-section{

padding:50px 20px;

}

.workspace-card{

position:relative;

top:0;

}

}

"""