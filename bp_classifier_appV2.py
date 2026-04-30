"""
app.py — Paediatric BP Centile Classifier
------------------------------------------
Classifies blood pressure in children aged 1–17 using the
ESH 2016 guidelines, with LMS-based height percentile calculation following UK-WHO reference data.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BP Centile Classifier", layout="centered")

st.markdown("""
<style>
  .result-card  { padding:1.6rem 2rem; border-radius:12px; margin:1.2rem 0; }
  .bp-normal    { background:#eafaf1; border-left:6px solid #27ae60; }
  .bp-highnorm  { background:#fef9e7; border-left:6px solid #f39c12; }
  .bp-stage1    { background:#fdf2e9; border-left:6px solid #e67e22; }
  .bp-stage2    { background:#fdedec; border-left:6px solid #e74c3c; }
  .bp-iso       { background:#eef2ff; border-left:6px solid #3949ab; }
  .bp-unknown   { background:#f4f6f7; border-left:6px solid #95a5a6; }
  .result-label { font-size:2rem; font-weight:700; margin:0; }
  .result-desc  { font-size:1rem; margin:0.4rem 0 0 0; color:#444; }
  .flag-text    { font-size:1rem; font-weight:600; color:#c0392b; margin-top:0.6rem; }
  .ref-badge    { font-size:0.8rem; color:#888; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Load reference data ───────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    who_g02  = pd.read_excel(os.path.join(APP_DIR, "WHO_2006.xlsx"),  sheet_name="girls_0_2")
    who_g25  = pd.read_excel(os.path.join(APP_DIR, "WHO_2006.xlsx"),  sheet_name="girls_2_5")
    who_b02  = pd.read_excel(os.path.join(APP_DIR, "WHO_2006.xlsx"),  sheet_name="boys_0_2")
    who_b25  = pd.read_excel(os.path.join(APP_DIR, "WHO_2006.xlsx"),  sheet_name="boys_2_5")
    uk90     = pd.read_excel(os.path.join(APP_DIR, "uk90_full.xlsx"), sheet_name="uk90_full")
    bp_table = pd.read_csv(os.path.join(APP_DIR, "esh2016_bp_table.csv"))

    for df in [who_g02, who_g25, who_b02, who_b25]:
        df["age"] = df["month"] / 12

    uk90 = uk90.rename(columns={"years": "age", "L.ht": "L", "M.ht": "M", "S.ht": "S"})
    uk90["sex_label"] = uk90["sex"].map({1: "M", 2: "F"})

    return who_g02, who_g25, who_b02, who_b25, uk90, bp_table

try:
    who_g02, who_g25, who_b02, who_b25, uk90, bp_table = load_data()
    table_ok = True
except FileNotFoundError as e:
    table_ok = False
    missing_file = str(e)


# ── Core functions ────────────────────────────────────────────────────────────
def calculate_age(dob: date, measurement_date: date):
    """Returns (years, months) as integers."""
    years  = measurement_date.year  - dob.year
    months = measurement_date.month - dob.month
    if measurement_date.day < dob.day:
        months -= 1
    if months < 0:
        years  -= 1
        months += 12
    return years, months


def height_to_percentile(height_cm, age_years, age_months, sex):
    """LMS interpolation -> snapped to nearest ESH reference percentile."""
    decimal_age = age_years + (age_months / 12.0)

    if sex == "M":
        if decimal_age <= 2:   df = who_b02
        elif decimal_age <= 4: df = who_b25
        else:                  df = uk90[uk90["sex_label"] == "M"]
    else:
        if decimal_age <= 2:   df = who_g02
        elif decimal_age <= 4: df = who_g25
        else:                  df = uk90[uk90["sex_label"] == "F"]

    L = np.interp(decimal_age, df["age"], df["L"])
    M = np.interp(decimal_age, df["age"], df["M"])
    S = np.interp(decimal_age, df["age"], df["S"])

    if L != 0:
        z = ((height_cm / M) ** L - 1) / (L * S)
    else:
        z = np.log(height_cm / M) / S

    raw_pct  = norm.cdf(z) * 100
    ESH_PCTS = [5, 10, 25, 50, 75, 90, 95]
    snapped  = min(ESH_PCTS, key=lambda p: abs(p - raw_pct))
    return snapped, raw_pct


def classify_bp(age_years, sex, height_pct, systolic, diastolic):
    """
    ESH 2016 BP classification.
    Returns (category_key, description, flag, thresholds_dict)
    """
    if not (1 <= age_years <= 17):
        return "unknown", "Age out of valid range (1-17 years).", False, {}

    # Age >=16: adult fixed thresholds (ESH 2016 Table 1)
    if age_years >= 16:
        if systolic < 130 and diastolic < 85:
            return "normal",   "Normal BP - no action required.", False, {}
        elif systolic < 140 and diastolic < 90:
            return "highnorm", "High-normal BP - monitor and recheck.", False, {}
        elif systolic >= 140 and diastolic < 90:
            return "iso",      "Isolated Systolic Hypertension - review clinically.", True, {}
        elif (160 <= systolic <= 179) or (100 <= diastolic <= 109):
            return "stage2",   "Stage 2 Hypertension - urgent clinical review.", True, {}
        elif (140 <= systolic <= 159) or (90 <= diastolic <= 99):
            return "stage1",   "Stage 1 Hypertension - further evaluation needed.", True, {}
        else:
            return "unknown",  "BP exceeds ESH 2016 table range - clinical review required.", True, {}

    # Age 0-15: percentile-based thresholds
    subset          = bp_table[bp_table["sex"] == sex]
    closest_age_row = subset.iloc[(subset["age"] - age_years).abs().argsort()[:1]]
    age_value       = closest_age_row["age"].values[0]
    age_subset      = subset[subset["age"] == age_value]
    row             = age_subset.iloc[
        (age_subset["height_percentile"] - height_pct).abs().argsort()[:1]
    ]

    s90 = row["sys_90"].values[0];  d90 = row["dia_90"].values[0]
    s95 = row["sys_95"].values[0];  d95 = row["dia_95"].values[0]
    s99 = row["sys_99"].values[0];  d99 = row["dia_99"].values[0]

    thresholds = {
        "90th":   (s90, d90),
        "95th":   (s95, d95),
        "99th+5": (s99 + 5, d99 + 5),
    }

    # ISH checked before Stage 1 — SBP >=95th with DBP <90th must not fall into Stage 1
    if systolic < s90 and diastolic < d90:
        return "normal",   "Normal BP - no action required.", False, thresholds
    elif (systolic >= s90 or diastolic >= d90) and (systolic < s95 and diastolic < d95):
        return "highnorm", "High-normal BP - monitor and recheck.", False, thresholds
    elif systolic >= s95 and diastolic < d90:
        return "iso",      "Isolated Systolic Hypertension - review clinically.", True, thresholds
    elif systolic > s99 + 5 or diastolic > d99 + 5:
        return "stage2",   "Stage 2 Hypertension - urgent clinical review.", True, thresholds
    elif systolic >= s95 or diastolic >= d95:
        return "stage1",   "Stage 1 Hypertension - further evaluation needed.", True, thresholds
    else:
        return "unknown",  "Unable to classify - please check values entered.", False, thresholds


RESULT_LABELS = {
    "normal":   "✅ Normal BP",
    "highnorm": "⚠️ High-Normal BP",
    "iso":      "⚠️ Isolated Systolic Hypertension",
    "stage1":   "🔴 Stage 1 Hypertension",
    "stage2":   "🔴 Stage 2 Hypertension",
    "unknown":  "❓ Unable to Classify",
}

CARD_CLASS = {
    "normal":   "bp-normal",
    "highnorm": "bp-highnorm",
    "iso":      "bp-iso",
    "stage1":   "bp-stage1",
    "stage2":   "bp-stage2",
    "unknown":  "bp-unknown",
}


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Paediatric BP Centile Classifier")
st.caption("Blood pressure classification for children aged 1-17  ·  ESH 2016 guidelines")
st.divider()

if not table_ok:
    st.error(
        f"Reference data files not found. Ensure WHO_2006.xlsx, uk90_full.xlsx, "
        f"and esh2016_bp_table.csv are in the same folder as app.py.\n\n{missing_file}"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.caption(
        "Classifies BP using the **ESH 2016 guidelines** "
        "(Lurbe et al., *J Hypertens* 2016).\n\n"
        "**Age 1-15:** percentile-based cutoffs (90th / 95th / 99th+5). "
        "Height percentile is calculated automatically from height in cm "
        "using WHO/UK90 LMS reference data.\n\n"
        "**Age 16-17:** adult-equivalent fixed cutoffs (130/85, 140/90)."
    )
    st.divider()
    st.caption("⚠️ For clinical decision support only. Always apply clinical judgement.")
    st.divider()
    st.caption("Group Project — Healthcare Technology")


# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Patient details")

col1, col2 = st.columns(2)
with col1:
    dob = st.date_input(
        "Date of birth",
        value=date(2017, 1, 1),
        min_value=date(2005, 1, 1),
        max_value=date.today(),
        help="Child's date of birth",
    )
with col2:
    meas_date = st.date_input(
        "Date of measurement",
        value=date.today(),
        max_value=date.today(),
        help="Date when BP was measured",
    )

sex = st.selectbox(
    "Sex",
    options=["M", "F"],
    format_func=lambda x: "Male" if x == "M" else "Female",
)

st.subheader("Measurements")
col3, col4, col5 = st.columns(3)
with col3:
    height_cm = st.number_input("Height (cm)",          min_value=50.0, max_value=220.0, value=120.0, step=0.5)
with col4:
    systolic  = st.number_input("Systolic BP (mmHg)",   min_value=50,   max_value=250,   value=110,   step=1)
with col5:
    diastolic = st.number_input("Diastolic BP (mmHg)",  min_value=30,   max_value=150,   value=70,    step=1)

st.divider()

# ── Classify button ───────────────────────────────────────────────────────────
classify_clicked = st.button("Classify BP", type="primary", use_container_width=True)

if classify_clicked:
    if meas_date < dob:
        st.error("Measurement date cannot be before date of birth.")
      elif systolic <= diastolic:
          st.error(
            f"Invalid BP reading: systolic ({systolic} mmHg) must be greater than "
            f"diastolic ({diastolic} mmHg). Please check the values entered."
        )
    elif systolic - diastolic < 10:
        st.error(
            f"Implausible BP reading: pulse pressure ({systolic - diastolic} mmHg) is "
            f"too narrow. Please check the values entered."
        )

  
    else:
        age_years, age_months = calculate_age(dob, meas_date)

        if not (1 <= age_years <= 17):
            st.warning("This tool is validated for children aged 1-17 years.")
        else:
            height_pct_snapped, height_pct_raw = height_to_percentile(
                height_cm, age_years, age_months, sex
            )

            category, description, flag, thresholds = classify_bp(
                age_years, sex, height_pct_snapped, systolic, diastolic
            )

            label     = RESULT_LABELS[category]
            card_cls  = CARD_CLASS[category]
            flag_html = '<p class="flag-text">&#x2691; Flag for clinical review</p>' if flag else ""

            st.markdown(f"""
            <div class="result-card {card_cls}">
              <p class="result-label">{label}</p>
              <p class="result-desc">{description}</p>
              {flag_html}
              <p class="ref-badge">
                Age {age_years}y {age_months}m &nbsp;&middot;&nbsp;
                {"Male" if sex == "M" else "Female"} &nbsp;&middot;&nbsp;
                Height {height_cm} cm ({height_pct_snapped}th %tile, raw {height_pct_raw:.1f}th) &nbsp;&middot;&nbsp;
                BP {systolic}/{diastolic} mmHg
              </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Reference thresholds expander ─────────────────────────────
            with st.expander("Reference thresholds for this patient"):
                if thresholds:
                    tbl = pd.DataFrame({
                        "Threshold":        ["90th percentile", "95th percentile", "99th + 5 mmHg"],
                        "Systolic (mmHg)":  [thresholds["90th"][0],   thresholds["95th"][0],   thresholds["99th+5"][0]],
                        "Diastolic (mmHg)": [thresholds["90th"][1],   thresholds["95th"][1],   thresholds["99th+5"][1]],
                    })
                    st.table(tbl.set_index("Threshold"))
                else:
                    st.markdown("""
| Category | Systolic (mmHg) | Diastolic (mmHg) |
|---|---|---|
| Normal | < 130 | < 85 |
| High-normal | 130–139 | 85–89 |
| Stage 1 HTN | 140–159 | 90–99 |
| Stage 2 HTN | 160–179 | 100–109 |
| ISH | ≥ 140 | < 90 |
                    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Reference: Lurbe E et al. 2016 ESH guidelines for management of high BP "
    "in children and adolescents. J Hypertens 2016;34:1887-1920."
)
