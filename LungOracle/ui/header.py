# ui/header.py

"""
Header component for LungOracle.

Provides the landing hero displayed at the top of the application.
"""

from datetime import datetime


def get_header():
    year = datetime.now().year

    return f"""
<div style="
background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
border-bottom:1px solid #e8eef5;
font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;
">

<!-- ======================= NAVIGATION ======================= -->

<div style="
max-width:1450px;
margin:auto;
height:78px;
display:flex;
justify-content:space-between;
align-items:center;
padding:0 42px;
">

<div style="
display:flex;
align-items:center;
gap:18px;
">

<div style="
width:54px;
height:54px;
border-radius:16px;
background:linear-gradient(135deg,#2563eb,#1d4ed8);
display:flex;
justify-content:center;
align-items:center;
font-size:26px;
color:white;
box-shadow:0 10px 24px rgba(37,99,235,.25);
">
🧬
</div>

<div>

<div style="
font-size:31px;
font-weight:800;
letter-spacing:-1px;
color:#1e293b;
line-height:1;
">

LungOracle

</div>

<div style="
margin-top:5px;
font-size:13px;
color:#64748b;
font-weight:500;
">

AI-powered Multi-Modal Molecular Survival Prediction

</div>

</div>

</div>



<div style="
display:flex;
gap:30px;
align-items:center;
">

<a href="https://github.com/ParthShringarpure09/Lung_adenocarcinoma_survival"
target="_blank"
style="
text-decoration:none;
font-size:14px;
font-weight:600;
color:#475569;
">

GitHub

</a>

<a href="#research"
style="
text-decoration:none;
font-size:14px;
font-weight:600;
color:#475569;
">

Research

</a>

<a href="#about"
style="
text-decoration:none;
font-size:14px;
font-weight:600;
color:#475569;
">

About

</a>

</div>

</div>





<!-- ======================= HERO ======================= -->

<div style="
max-width:1450px;
margin:auto;
padding:60px 42px 58px;
display:flex;
justify-content:space-between;
align-items:center;
gap:70px;
">

<div style="
flex:1;
max-width:760px;
">

<div style="
display:inline-flex;
align-items:center;
gap:8px;
padding:8px 15px;
background:#eff6ff;
border-radius:999px;
font-size:12px;
font-weight:700;
letter-spacing:.5px;
color:#2563eb;
margin-bottom:24px;
">

SURVIVAL AI

</div>

<h1 style="
margin:0;
font-size:50px;
line-height:1.12;
font-weight:700;
letter-spacing:-1.8px;
color:#1e293b;
">

Predicting Lung Adenocarcinoma Survival

<br>

<span style="color:#2563eb;">

Beyond Clinical Staging

</span>

</h1>

<div style="
margin-top:24px;
font-size:18px;
line-height:1.9;
color:#64748b;
max-width:690px;
">

LungOracle integrates Gene expression, tumour dysregulation,
immune microenvironment and clinical variables into a
 ensemble machine learning framework for
personalised survival prediction.

</div>

<div style="
margin-top:34px;
display:flex;
gap:12px;
flex-wrap:wrap;
">

<div class="lo-badge">
✓  Pipeline
</div>

<div class="lo-badge">
✓ Explainable AI
</div>

<div class="lo-badge">
✓ External Validation
</div>

</div>

</div>





<div style="
width:430px;
display:grid;
grid-template-columns:1fr 1fr;
gap:18px;
">

<div class="lo-card lo-metric">
<div class="lo-metric-value">

0.702

</div>
<div class="lo-metric-label">

 C-index

</div>
</div>

<div class="lo-card lo-metric">
<div class="lo-metric-value">

478

</div>
<div class="lo-metric-label">

TCGA Patients

</div>
</div>

<div class="lo-card lo-metric">
<div class="lo-metric-value">

124

</div>
<div class="lo-metric-label">

Integrated Features

</div>
</div>

<div class="lo-card lo-metric">
<div class="lo-metric-value">

4

</div>
<div class="lo-metric-label">

Validation Cohorts

</div>
</div>

</div>

</div>





<div style="
max-width:1450px;
margin:auto;
padding:0 42px 18px;
">

<div style="
font-size:11px;
color:#94a3b8;
text-align:right;
">

© {year} LungOracle 

</div>

</div>

</div>
"""