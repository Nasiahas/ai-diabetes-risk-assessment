"""
AI Diabetes Risk Assessment
Rocheston Certified AI Engineer (RCAI) classroom project.

Loads the pre-trained KNN model (Diabetesmodel.pkl) and the MinMaxScaler
(scaler.pkl) produced by Diabetes_prediction.ipynb, then predicts diabetes
risk from four inputs: Glucose, Insulin, BMI and Age.

Pages (sidebar): Welcome, Assessment, Results & report, Prevention guide,
About the model.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ===========================================================================
# 1. SETTINGS
# ===========================================================================
st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Diabetesmodel.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "diabetes.csv"

# The scaler was fitted on all 9 dataset columns, in this order:
#   0 Pregnancies, 1 Glucose, 2 BloodPressure, 3 SkinThickness, 4 Insulin,
#   5 BMI, 6 DiabetesPedigreeFunction, 7 Age, 8 Outcome
# The model was trained on columns 1, 4, 5 and 7 only (notebook:
# X = dataset_scaled.iloc[:, [1, 4, 5, 7]]). Order matters.
FEATURES = ["Glucose", "Insulin", "BMI", "Age"]
FEATURE_COLUMNS = [1, 4, 5, 7]

# Accuracy scores from the course material. Replace them with the values
# printed in your own notebook if they differ.
REPORTED_ACCURACY = {
    "Logistic Regression": 78,
    "KNN": 76,
    "Random Forest": 79,
    "Decision Tree": 75,
}

# Guardrails: values outside these limits are rejected (no prediction).
HARD_LIMITS = {
    "Glucose": (40, 500),   # mg/dL
    "Insulin": (2, 900),    # μU/mL
    "BMI": (10, 80),        # kg/m²
    "Age": (21, 120),       # years; the dataset only contains adults 21+
}
UNITS = {"Glucose": "mg/dL", "Insulin": "μU/mL", "BMI": "kg/m²", "Age": "years"}

GLUCOSE_TESTS = [
    "2-hour glucose test (OGTT)",
    "Fasting glucose",
    "Random glucose",
]

CUSTOM = "Custom (enter my own values)"
PRESETS = {
    "Example: low-risk adult": dict(
        glucose=90.0, insulin=70.0, bmi=22.5, age=26, systolic=115, diastolic=75),
    "Example: borderline adult": dict(
        glucose=135.0, insulin=150.0, bmi=29.0, age=42, systolic=128, diastolic=82),
    "Example: high-risk adult": dict(
        glucose=180.0, insulin=220.0, bmi=38.0, age=55, systolic=145, diastolic=92),
}

PAGES = [
    "🏠 Welcome",
    "🩺 Assessment",
    "📋 Results & report",
    "🌱 Prevention guide",
    "📚 About the model",
]

# ===========================================================================
# 2. SESSION STATE (remembers inputs and results between pages)
# ===========================================================================
INPUT_DEFAULTS = {
    "preset": CUSTOM,
    "glucose_test": GLUCOSE_TESTS[0],
    "glucose": None,
    "insulin": None,
    "bmi": None,
    "age": None,
    "use_bmi_calc": False,
    "height_cm": None,
    "weight_kg": None,
    "use_bp": False,
    "systolic": None,
    "diastolic": None,
}

for key, default in INPUT_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default
    else:
        # Re-saving keeps the value when its page isn't showing.
        st.session_state[key] = st.session_state[key]

st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("result", None)


def go_to(page):
    st.session_state.page = page


def apply_preset():
    name = st.session_state.preset
    if name in PRESETS:
        for key, value in PRESETS[name].items():
            st.session_state[key] = value
        st.session_state.use_bmi_calc = False


def clear_form():
    for key, default in INPUT_DEFAULTS.items():
        st.session_state[key] = default


# ===========================================================================
# 3. LOADING THE MODEL, SCALER AND DATASET
# ===========================================================================
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def training_ranges(scaler):
    """Smallest and largest value of each feature the model saw in training."""
    return {
        name: (float(scaler.data_min_[col]), float(scaler.data_max_[col]))
        for name, col in zip(FEATURES, FEATURE_COLUMNS)
    }


def predict(model, scaler, values):
    """Scale the four raw inputs exactly as the notebook did, then predict."""
    raw = np.array(values, dtype=float)
    scaled = raw * scaler.scale_[FEATURE_COLUMNS] + scaler.min_[FEATURE_COLUMNS]
    X = scaled.reshape(1, -1)
    prediction = int(round(float(model.predict(X)[0])))
    risk = None
    if hasattr(model, "predict_proba"):
        classes = [int(round(float(c))) for c in model.classes_]
        risk = float(model.predict_proba(X)[0][classes.index(1)])
    return prediction, risk


# ===========================================================================
# 4. VALIDATION (guardrails)
# ===========================================================================
def validate(inputs, bp, ranges):
    """Return (errors, cautions). Errors block the prediction; cautions don't."""
    errors, cautions = [], []

    missing = [name for name in FEATURES if inputs[name] is None]
    if missing:
        errors.append("Fill in every measurement. Missing: " + ", ".join(missing) + ".")
        return errors, cautions

    g, ins, bmi, age = (inputs[name] for name in FEATURES)

    if g < HARD_LIMITS["Glucose"][0]:
        errors.append(
            f"Glucose of {g:g} mg/dL is below the accepted range (40–500 mg/dL). "
            "If your reading is in mmol/L, multiply it by 18 to convert "
            "(for example, 6.5 mmol/L ≈ 117 mg/dL). A real glucose this low "
            "is dangerous: seek medical help right away."
        )
    elif g > HARD_LIMITS["Glucose"][1]:
        errors.append(f"Glucose of {g:g} mg/dL is above the accepted range (40–500 mg/dL). Check the value.")

    lo, hi = HARD_LIMITS["Insulin"]
    if not lo <= ins <= hi:
        errors.append(f"Insulin of {ins:g} μU/mL is outside the accepted range ({lo}–{hi} μU/mL). Check the value.")

    lo, hi = HARD_LIMITS["BMI"]
    if not lo <= bmi <= hi:
        errors.append(f"BMI of {bmi:.1f} is outside the accepted range ({lo}–{hi}). Check the value, or use the BMI calculator.")

    if age < HARD_LIMITS["Age"][0]:
        errors.append(
            f"Age {age} is below 21. The model was trained only on adults aged 21 "
            "and over, so it can't give a reliable result for younger people."
        )
    elif age > HARD_LIMITS["Age"][1]:
        errors.append(f"Age {age} is above the accepted range (21–120). Check the value.")

    if bp is not None:
        sys_bp, dia_bp = bp
        if sys_bp is None or dia_bp is None:
            errors.append("Enter both systolic and diastolic blood pressure, or switch blood pressure off.")
        else:
            if not 70 <= sys_bp <= 250:
                errors.append(f"Systolic pressure of {sys_bp} mmHg is outside the accepted range (70–250).")
            if not 40 <= dia_bp <= 150:
                errors.append(f"Diastolic pressure of {dia_bp} mmHg is outside the accepted range (40–150).")
            if sys_bp <= dia_bp:
                errors.append("Systolic pressure (top number) must be higher than diastolic (bottom number).")

    if not errors:
        for name in FEATURES:
            low, high = ranges[name]
            value = inputs[name]
            outside_low = low > 0 and value < low
            if outside_low or value > high:
                cautions.append(
                    f"{name} ({value:g} {UNITS[name]}) is outside the range seen in "
                    f"the training data ({low:g}–{high:g}). Treat this result with extra caution."
                )
    return errors, cautions


# ===========================================================================
# 5. CLINICAL REFERENCE CATEGORIES
# ===========================================================================
ADVICE = {
    "ok": "Within the usual range. Keep up healthy habits and recheck as your doctor advises.",
    "warn": "Worth discussing with a healthcare professional, who may suggest a repeat test and lifestyle changes.",
    "high": "Arrange a follow-up with a healthcare professional. A diagnosis needs a confirmed repeat test.",
}


def assess_glucose(value, test):
    if value < 70:
        return dict(
            label="Below usual range", level="warn",
            reference="Below 70 mg/dL is considered low (hypoglycaemia).",
            advice="If this is a real reading, especially with shakiness, sweating or confusion, contact a healthcare professional.",
        )
    if test == "Fasting glucose":
        reference = "Fasting: under 100 normal; 100–125 prediabetes; 126 or more diabetes range (ADA)."
        bands = [(100, "Normal", "ok"), (126, "Prediabetes range", "warn"), (float("inf"), "Diabetes range", "high")]
    elif test == "Random glucose":
        reference = ("Random: 200 or more with symptoms of high blood sugar meets diagnostic "
                     "criteria (ADA). Lower values need a fasting or 2-hour test to interpret.")
        bands = [(140, "Typical", "ok"), (200, "Above typical", "warn"), (float("inf"), "High", "high")]
    else:
        reference = "2-hour OGTT: under 140 normal; 140–199 prediabetes; 200 or more diabetes range (ADA)."
        bands = [(140, "Normal", "ok"), (200, "Prediabetes range", "warn"), (float("inf"), "Diabetes range", "high")]
    for limit, label, level in bands:
        if value < limit:
            return dict(label=label, level=level, reference=reference, advice=ADVICE[level])


def assess_bmi(value):
    reference = "WHO adult BMI: under 18.5 underweight; 18.5–24.9 healthy; 25–29.9 overweight; 30 or more obesity."
    if value < 18.5:
        return dict(label="Underweight", level="warn", reference=reference,
                    advice="Discuss nutrition and healthy weight gain with a healthcare professional.")
    if value < 25:
        return dict(label="Healthy range", level="ok", reference=reference, advice=ADVICE["ok"])
    advice = ("Losing 5–7% of body weight, with regular activity, cut type 2 diabetes "
              "risk by 58% in the Diabetes Prevention Program trial.")
    if value < 30:
        return dict(label="Overweight", level="warn", reference=reference, advice=advice)
    return dict(label="Obesity range", level="high", reference=reference, advice=advice)


def assess_insulin(value):
    return dict(
        label="Context-dependent", level="info",
        reference=("The dataset uses 2-hour serum insulin. A single insulin value can't be "
                   "labelled good or bad without the test conditions and your glucose result."),
        advice="Ask the laboratory or your doctor to interpret it alongside your glucose.",
    )


def assess_age(value):
    reference = "ADA recommends diabetes screening for all adults from age 35, or earlier with risk factors such as overweight."
    if value < 35:
        return dict(label="Under routine screening age", level="info", reference=reference,
                    advice="Screening may still be worthwhile if you have other risk factors.")
    return dict(label="Routine screening age", level="info", reference=reference,
                advice="If you haven't had a diabetes test in the last 3 years, ask about one.")


def assess_bp(systolic, diastolic):
    reference = "ACC/AHA 2017: normal under 120/80; elevated 120–129/under 80; stage 1 130–139 or 80–89; stage 2 140+ or 90+."
    if systolic > 180 or diastolic > 120:
        return dict(label="Hypertensive crisis range", level="high", reference=reference,
                    advice="Seek medical care promptly, especially if you have chest pain, breathlessness or headache.")
    if systolic >= 140 or diastolic >= 90:
        return dict(label="Stage 2 hypertension range", level="high", reference=reference, advice=ADVICE["high"])
    if systolic >= 130 or diastolic >= 80:
        return dict(label="Stage 1 hypertension range", level="warn", reference=reference, advice=ADVICE["warn"])
    if systolic >= 120:
        return dict(label="Elevated", level="warn", reference=reference, advice=ADVICE["warn"])
    return dict(label="Normal", level="ok", reference=reference, advice=ADVICE["ok"])


def badge(level, label):
    icon = {"ok": "🟢", "warn": "🟠", "high": "🔴", "info": "🔵"}[level]
    return f"{icon} **{label}**"


def guidance_card(title, value_text, info):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(badge(info["level"], info["label"]))
        st.markdown(f"### {value_text}")
        st.caption(info["reference"])
        st.write(info["advice"])


# ===========================================================================
# 6. DOWNLOADABLE REPORT
# ===========================================================================
def build_report_text(r):
    i = r["inputs"]
    lines = [
        "AI DIABETES RISK ASSESSMENT REPORT",
        "=" * 44,
        f"Generated: {r['time']}",
        "",
        "RESULT",
        f"  {r['headline']}",
    ]
    if r["risk"] is not None:
        lines.append(f"  Estimated risk: {r['risk']:.0%} (share of the most similar training patients who had diabetes)")
    lines += ["", "MEASUREMENTS"]
    lines.append(f"  Glucose: {i['Glucose']:g} mg/dL ({r['glucose_test']}) - {r['guidance']['Glucose']['label']}")
    lines.append(f"  Insulin: {i['Insulin']:g} μU/mL - {r['guidance']['Insulin']['label']}")
    lines.append(f"  BMI: {i['BMI']:.1f} ({r['bmi_source']}) - {r['guidance']['BMI']['label']}")
    lines.append(f"  Age: {i['Age']} years - {r['guidance']['Age']['label']}")
    if r["bp"] is not None:
        lines.append(f"  Blood pressure: {r['bp'][0]}/{r['bp'][1]} mmHg - {r['guidance']['BP']['label']} (not used by the model)")
    if r["cautions"]:
        lines += ["", "RELIABILITY NOTES"] + [f"  - {c}" for c in r["cautions"]]
    lines += ["", "GUIDANCE"]
    for name, g in r["guidance"].items():
        lines.append(f"  {name}: {g['advice']}")
    lines += [
        "",
        "MODEL",
        "  K-Nearest Neighbors trained on the Pima Indians Diabetes dataset",
        "  (768 women aged 21+). Inputs: Glucose, Insulin, BMI, Age.",
        "",
        "DISCLAIMER",
        "  Educational tool only. This is not a medical diagnosis. Consult a",
        "  qualified healthcare professional about any health concerns.",
    ]
    return "\n".join(lines)


def build_report_csv(r):
    i = r["inputs"]
    row = {
        "generated": r["time"],
        "glucose_mg_dl": i["Glucose"],
        "glucose_test": r["glucose_test"],
        "insulin_uU_ml": i["Insulin"],
        "bmi": round(i["BMI"], 1),
        "age": i["Age"],
        "systolic_mmhg": r["bp"][0] if r["bp"] else None,
        "diastolic_mmhg": r["bp"][1] if r["bp"] else None,
        "prediction": "likely diabetic" if r["prediction"] == 1 else "not likely diabetic",
        "estimated_risk": round(r["risk"], 3) if r["risk"] is not None else None,
    }
    return pd.DataFrame([row]).to_csv(index=False)


# ===========================================================================
# 7. PAGES
# ===========================================================================
def page_welcome():
    st.title("🩺 AI Diabetes Risk Assessment")
    st.write(
        "Estimate the likelihood of diabetes from four routine health measurements, "
        "then see what each number means and what you can do next."
    )

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("How it works")
            st.markdown(
                "1. **Enter your measurements** — glucose, insulin, BMI and age. "
                "You can calculate BMI from height and weight, or try an example patient.\n"
                "2. **Get your assessment** — a machine-learning model compares you with "
                "768 patients whose outcomes are known.\n"
                "3. **Review and download** — see guidance for each measurement and save a report."
            )
            st.button("Start my assessment", type="primary",
                      on_click=go_to, args=(PAGES[1],))
        with st.container(border=True):
            st.subheader("What you'll need")
            st.markdown(
                "- A **glucose** reading in mg/dL (ideally from a 2-hour glucose test)\n"
                "- An **insulin** reading in μU/mL from a lab test\n"
                "- Your **BMI**, or your height and weight\n"
                "- Your **age** (21 or older)\n"
                "- Optional: your **blood pressure**, for the report"
            )
    with col2:
        with st.container(border=True):
            st.subheader("Who it's for")
            st.write(
                "Adults aged 21 and over who want to understand their diabetes risk, "
                "and students exploring how AI supports healthcare decisions."
            )
        st.warning(
            "**This is an educational tool, not a medical diagnosis.** Diabetes can only be "
            "diagnosed by a healthcare professional using confirmed laboratory tests."
        )
        st.info("New to these measurements? See **📚 About the model** for the dataset, "
                "how the model works and its limitations.")


def page_assessment(model, scaler, ranges):
    st.title("🩺 Assessment")
    st.write("Enter your four measurements. All four are required; blood pressure is optional.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Your measurements")
            st.selectbox(
                "Start from an example patient (optional)",
                [CUSTOM] + list(PRESETS), key="preset", on_change=apply_preset,
                help="Fills the form with sample values so you can see how the app works.",
            )
            st.radio(
                "Type of glucose test", GLUCOSE_TESTS, key="glucose_test", horizontal=True,
                help="Used for the clinical reference ranges. The model was trained on 2-hour test values.",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Glucose (mg/dL)", min_value=0.0, max_value=1000.0, step=1.0,
                    key="glucose", placeholder="e.g. 120",
                    help="Plasma glucose in mg/dL. Accepted range 40–500. From mmol/L, multiply by 18.",
                )
            with c2:
                st.number_input(
                    "Insulin (μU/mL)", min_value=0.0, max_value=2000.0, step=1.0,
                    key="insulin", placeholder="e.g. 80",
                    help="2-hour serum insulin from a lab test. Accepted range 2–900.",
                )

            st.toggle("Calculate BMI from height and weight", key="use_bmi_calc")
            c3, c4 = st.columns(2)
            bmi_value, bmi_source = None, "entered"
            if st.session_state.use_bmi_calc:
                with c3:
                    st.number_input("Height (cm)", min_value=0.0, max_value=300.0, step=1.0,
                                    key="height_cm", placeholder="e.g. 165")
                with c4:
                    st.number_input("Weight (kg)", min_value=0.0, max_value=400.0, step=0.5,
                                    key="weight_kg", placeholder="e.g. 68")
                h, w = st.session_state.height_cm, st.session_state.weight_kg
                if h and w and 100 <= h <= 250 and 20 <= w <= 350:
                    bmi_value = w / (h / 100) ** 2
                    bmi_source = "calculated from height and weight"
                    st.info(f"Calculated BMI: **{bmi_value:.1f}**")
                elif h or w:
                    st.caption("Enter a height of 100–250 cm and a weight of 20–350 kg.")
            else:
                with c3:
                    st.number_input("BMI (kg/m²)", min_value=0.0, max_value=200.0, step=0.1,
                                    format="%.1f", key="bmi", placeholder="e.g. 25.0",
                                    help="Body mass index. Accepted range 10–80.")
                bmi_value = st.session_state.bmi

            with (c4 if not st.session_state.use_bmi_calc else st.container()):
                st.number_input("Age (years)", min_value=0, max_value=150, step=1,
                                key="age", placeholder="e.g. 30",
                                help="Age in whole years. The model accepts 21–120.")

            st.toggle("Add blood pressure to my report", key="use_bp")
            bp = None
            if st.session_state.use_bp:
                c5, c6 = st.columns(2)
                with c5:
                    st.number_input("Systolic (top number, mmHg)", min_value=0, max_value=300,
                                    step=1, key="systolic", placeholder="e.g. 120")
                with c6:
                    st.number_input("Diastolic (bottom number, mmHg)", min_value=0, max_value=200,
                                    step=1, key="diastolic", placeholder="e.g. 80")
                st.caption("Blood pressure appears in your report with its clinical category. "
                           "The AI model doesn't use it.")
                bp = (st.session_state.systolic, st.session_state.diastolic)

            b1, b2 = st.columns([3, 1])
            assess = b1.button("Assess my risk", type="primary", use_container_width=True)
            b2.button("Clear form", on_click=clear_form, use_container_width=True)

        if assess:
            inputs = {
                "Glucose": st.session_state.glucose,
                "Insulin": st.session_state.insulin,
                "BMI": bmi_value,
                "Age": st.session_state.age,
            }
            if st.session_state.use_bmi_calc and bmi_value is None:
                st.error("Enter a valid height and weight to calculate BMI.")
            else:
                errors, cautions = validate(inputs, bp, ranges)
                if errors:
                    for message in errors:
                        st.error(message)
                    st.session_state.result = None
                else:
                    prediction, risk = predict(model, scaler, [inputs[n] for n in FEATURES])
                    guidance = {
                        "Glucose": assess_glucose(inputs["Glucose"], st.session_state.glucose_test),
                        "Insulin": assess_insulin(inputs["Insulin"]),
                        "BMI": assess_bmi(inputs["BMI"]),
                        "Age": assess_age(inputs["Age"]),
                    }
                    if bp is not None:
                        guidance["BP"] = assess_bp(*bp)
                    st.session_state.result = dict(
                        inputs=inputs, bp=bp, glucose_test=st.session_state.glucose_test,
                        bmi_source=bmi_source, prediction=prediction, risk=risk,
                        cautions=cautions, guidance=guidance,
                        headline=("You are likely to have diabetes." if prediction == 1
                                  else "You are not likely to have diabetes."),
                        time=datetime.now().strftime("%d %b %Y, %H:%M"),
                    )
                    st.toast("Your assessment is ready.", icon="✅")

        result = st.session_state.result
        if result:
            with st.container(border=True):
                st.subheader("Your assessment")
                if result["prediction"] == 1:
                    st.error(f"⚠️ **{result['headline']}**")
                else:
                    st.success(f"✅ **{result['headline']}**")
                if result["risk"] is not None:
                    st.progress(result["risk"], text=f"Estimated risk: {result['risk']:.0%}")
                for caution in result["cautions"]:
                    st.warning(caution)
                st.caption(f"Assessed {result['time']}")
                st.button("See full results and download report", type="primary",
                          on_click=go_to, args=(PAGES[2],))

    with right:
        with st.container(border=True):
            st.subheader("Understanding your inputs")
            st.markdown(
                "**🩸 Glucose** — the main sugar in your blood. High levels over time "
                "are the defining feature of diabetes.\n\n"
                "**💉 Insulin** — the hormone that moves glucose into your cells. "
                "It's measured with a lab test.\n\n"
                "**⚖️ BMI** — weight relative to height. Excess weight is one of the "
                "strongest risk factors for type 2 diabetes.\n\n"
                "**🎂 Age** — risk rises with age, which is why screening is advised from 35."
            )
        with st.container(border=True):
            st.subheader("Accepted ranges")
            low_high = pd.DataFrame(
                {
                    "Measurement": FEATURES,
                    "Accepted": [f"{lo}–{hi} {UNITS[n]}" for n, (lo, hi) in HARD_LIMITS.items()],
                    "Seen in training": [
                        f"{ranges[n][0]:g}–{ranges[n][1]:g}" for n in FEATURES
                    ],
                }
            )
            st.dataframe(low_high, hide_index=True, use_container_width=True)
            st.caption("Values outside the training range are accepted with a reliability warning.")


def page_results():
    st.title("📋 Results & report")
    result = st.session_state.result
    if not result:
        st.info("No assessment yet. Complete the assessment to see your results here.")
        st.button("Go to assessment", type="primary", on_click=go_to, args=(PAGES[1],))
        return

    i = result["inputs"]
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("Your result")
            if result["prediction"] == 1:
                st.error(f"⚠️ **{result['headline']}**")
                st.write("The model found that most patients with similar measurements had diabetes. "
                         "Please arrange a check-up with a healthcare professional.")
            else:
                st.success(f"✅ **{result['headline']}**")
                st.write("The model found that most patients with similar measurements did not have "
                         "diabetes. Keep up healthy habits and regular check-ups.")
            if result["risk"] is not None:
                st.progress(result["risk"], text=f"Estimated risk: {result['risk']:.0%}")
                st.caption("The share of the most similar patients in the training data who had diabetes.")
            st.caption(f"Assessed {result['time']}")
    with col2:
        with st.container(border=True):
            st.subheader("Your measurements")
            m1, m2 = st.columns(2)
            m1.metric("Glucose", f"{i['Glucose']:g} mg/dL")
            m2.metric("Insulin", f"{i['Insulin']:g} μU/mL")
            m3, m4 = st.columns(2)
            m3.metric("BMI", f"{i['BMI']:.1f}")
            m4.metric("Age", f"{i['Age']} yrs")
            if result["bp"] is not None:
                st.metric("Blood pressure", f"{result['bp'][0]}/{result['bp'][1]} mmHg")

    for caution in result["cautions"]:
        st.warning(caution)

    st.subheader("Measurement-by-measurement guidance")
    g = result["guidance"]
    cards = [
        ("Glucose", f"{i['Glucose']:g} mg/dL", g["Glucose"]),
        ("Insulin", f"{i['Insulin']:g} μU/mL", g["Insulin"]),
        ("BMI", f"{i['BMI']:.1f}", g["BMI"]),
        ("Age", f"{i['Age']} years", g["Age"]),
    ]
    if "BP" in g:
        cards.append(("Blood pressure", f"{result['bp'][0]}/{result['bp'][1]} mmHg", g["BP"]))
    for start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, card in zip(cols, cards[start:start + 3]):
            with col:
                guidance_card(*card)

    st.subheader("Download your report")
    d1, d2 = st.columns(2)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    d1.download_button("Download report (.txt)", build_report_text(result),
                       file_name=f"diabetes_risk_report_{stamp}.txt", mime="text/plain",
                       use_container_width=True)
    d2.download_button("Download data (.csv)", build_report_csv(result),
                       file_name=f"diabetes_risk_data_{stamp}.csv", mime="text/csv",
                       use_container_width=True)
    st.caption("The text report is easy to read or print; the CSV opens in Excel.")


def page_prevention():
    st.title("🌱 Prevention guide")
    st.write(
        "Type 2 diabetes is often preventable. In the Diabetes Prevention Program trial, "
        "people at high risk who lost 5–7% of their body weight and exercised 150 minutes "
        "a week cut their risk of developing diabetes by 58%."
    )

    tips = [
        ("🥗 Eat well", "Fill half your plate with vegetables. Choose whole grains, beans and lean "
                       "protein. Cut back on sugary drinks, white bread and processed snacks."),
        ("🚶 Move more", "Aim for 150 minutes of moderate activity a week, such as brisk walking "
                        "for 30 minutes, 5 days a week. Break up long periods of sitting."),
        ("⚖️ Reach a healthy weight", "If you're overweight, losing even 5–7% of your weight makes "
                                     "a real difference. Small, steady changes last longer."),
        ("😴 Sleep and stress", "Aim for 7–9 hours of sleep. Poor sleep and long-term stress both "
                               "affect blood sugar control."),
        ("🚭 Avoid smoking, limit alcohol", "Smoking raises the risk of type 2 diabetes. If you "
                                           "drink, keep it moderate."),
        ("🩺 Get screened", "Adults should be screened from age 35, or earlier with risk factors "
                           "such as overweight, family history or high blood pressure."),
    ]
    for start in range(0, len(tips), 3):
        cols = st.columns(3)
        for col, (title, text) in zip(cols, tips[start:start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    st.write(text)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.error(
            "**See a doctor if you notice:** constant thirst, frequent urination, unexplained "
            "weight loss, blurred vision, extreme tiredness, or cuts that heal slowly."
        )
    with col2:
        with st.container(border=True):
            st.markdown("#### Know your numbers")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Measure": ["Fasting glucose", "2-hour glucose (OGTT)", "BMI", "Blood pressure"],
                        "Healthy target": ["Under 100 mg/dL", "Under 140 mg/dL", "18.5–24.9", "Under 120/80 mmHg"],
                    }
                ),
                hide_index=True, use_container_width=True,
            )


def page_about(model, scaler, ranges):
    st.title("📚 About the model")
    st.write("How the app works, the data behind it, and what it can and can't tell you.")

    tabs = st.tabs(["Dataset", "Model & output", "How it works",
                    "Assumptions & limitations", "Technical notes", "Clinical references"])

    # --- Dataset ---------------------------------------------------------
    with tabs[0]:
        df = load_dataset()
        st.write(
            "The model was trained on the **Pima Indians Diabetes dataset**, collected by the "
            "US National Institute of Diabetes and Digestive and Kidney Diseases. Every record "
            "is a woman of Pima heritage aged 21 or over."
        )
        if df is None:
            st.info("Add `diabetes.csv` to the app folder to preview the dataset here.")
        else:
            diabetic = int(df["Outcome"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Patient records", len(df))
            m2.metric("Columns", df.shape[1])
            m3.metric("With diabetes", f"{diabetic} ({diabetic / len(df):.0%})")
            st.markdown("**First 20 records**")
            st.dataframe(df.head(20), use_container_width=True)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Outcome balance**")
                counts = df["Outcome"].map({0: "No diabetes", 1: "Diabetes"}).value_counts()
                st.bar_chart(counts)
            with c2:
                st.markdown("**The four features used by the model**")
                st.dataframe(df[FEATURES].describe().round(1), use_container_width=True)
            st.caption("Zeros in columns such as Glucose, Insulin and BMI usually mean the "
                       "measurement was missing in the original data.")

    # --- Model & output --------------------------------------------------
    with tabs[1]:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("#### Model used")
                details = {"Algorithm": type(model).__name__}
                for attr, label in [("n_neighbors", "Neighbours compared (k)"),
                                    ("weights", "Neighbour weighting"),
                                    ("metric", "Distance metric")]:
                    if hasattr(model, attr):
                        details[label] = str(getattr(model, attr))
                details["Inputs"] = ", ".join(FEATURES)
                details["Output"] = "0 = not diabetic, 1 = diabetic, plus a risk probability"
                st.dataframe(pd.DataFrame(details.items(), columns=["Property", "Value"]),
                             hide_index=True, use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("#### Accuracy of the four algorithms tested")
                st.bar_chart(pd.Series(REPORTED_ACCURACY, name="Accuracy (%)"))
                st.caption("KNN was saved to Diabetesmodel.pkl in the project notebook, although "
                           "Random Forest scored slightly higher.")

        st.markdown("#### Example outputs")
        st.write("The example patients from the assessment page, run through the model:")
        rows = []
        for name, p in PRESETS.items():
            pred, risk = predict(model, scaler, [p["glucose"], p["insulin"], p["bmi"], p["age"]])
            rows.append({
                "Patient": name.replace("Example: ", ""),
                "Glucose": p["glucose"], "Insulin": p["insulin"], "BMI": p["bmi"], "Age": p["age"],
                "Prediction": "Likely diabetic" if pred == 1 else "Not likely diabetic",
                "Estimated risk": f"{risk:.0%}" if risk is not None else "n/a",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- How it works ----------------------------------------------------
    with tabs[2]:
        st.markdown(
            "1. **Check the inputs.** Guardrails reject impossible values (for example glucose "
            "below 40 mg/dL or age under 21) and warn when a value is outside the training data.\n"
            "2. **Scale the inputs.** Each value is squashed into the 0–1 range using the same "
            "MinMaxScaler fitted during training, so all four measurements count equally.\n"
            "3. **Find similar patients.** KNN measures the distance between you and every patient "
            "in the training data, and picks the closest ones.\n"
            "4. **Vote.** If most of those neighbours had diabetes, the result is *likely*; "
            "otherwise *not likely*. The share who had diabetes is shown as the estimated risk."
        )
        st.info("Why scaling matters: without it, insulin (values up to 846) would swamp BMI "
                "(values up to 67) when KNN measures distances.")

    # --- Assumptions & limitations -------------------------------------
    with tabs[3]:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Assumptions")
            st.markdown(
                "- Measurements are accurate and use the stated units (mg/dL, μU/mL).\n"
                "- Glucose and insulin come from a 2-hour glucose tolerance test, as in the dataset.\n"
                "- The user is an adult aged 21 or over.\n"
                "- Patients similar to the training population have similar outcomes."
            )
        with c2:
            st.markdown("#### Limitations")
            st.markdown(
                "- **Narrow population:** trained only on Pima women aged 21+, so results may be "
                "less accurate for men, other ethnic groups and different age ranges.\n"
                "- **Small dataset:** 768 records, about 35% with diabetes (imbalanced).\n"
                "- **Moderate accuracy:** around 76% for KNN, so roughly 1 in 4 results may be wrong.\n"
                "- **Only four inputs:** family history, blood pressure, HbA1c and lifestyle are not "
                "considered.\n"
                "- **Not a diagnosis:** diabetes is diagnosed with confirmed lab tests such as HbA1c, "
                "fasting glucose or an OGTT.\n"
                "- No subgroup fairness analysis was performed."
            )

    # --- Technical notes -------------------------------------------------
    with tabs[4]:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Software")
            st.dataframe(
                pd.DataFrame({
                    "Component": ["Python app framework", "Machine learning", "Numerical", "Data"],
                    "Library": ["Streamlit", "scikit-learn", "NumPy", "pandas"],
                    "Version": [st.__version__, sklearn.__version__, np.__version__, pd.__version__],
                }),
                hide_index=True, use_container_width=True,
            )
            st.markdown("#### Files")
            st.markdown(
                "- `diabetes_app.py` — this app\n"
                "- `Diabetesmodel.pkl` — trained KNN model\n"
                "- `scaler.pkl` — MinMaxScaler fitted on the 9 dataset columns\n"
                "- `diabetes.csv` — dataset (for the preview above)\n"
                "- `requirements.txt` — library versions for deployment"
            )
        with c2:
            st.markdown("#### Scaling formula")
            st.latex(r"x_{scaled} = x \times scale_{col} + min_{col}")
            st.caption("Applied only to columns 1, 4, 5 and 7 (Glucose, Insulin, BMI, Age).")
            st.markdown("#### Validation rules")
            st.dataframe(
                pd.DataFrame({
                    "Measurement": FEATURES,
                    "Rejected outside": [f"{lo}–{hi}" for lo, hi in HARD_LIMITS.values()],
                    "Warning outside (training range)": [
                        f"{ranges[n][0]:g}–{ranges[n][1]:g}" for n in FEATURES],
                }),
                hide_index=True, use_container_width=True,
            )
            st.caption("The model and scaler load once and are cached, so predictions are instant.")

    # --- Clinical references ---------------------------------------------
    with tabs[5]:
        st.markdown(
            "1. American Diabetes Association. *Standards of Care in Diabetes: "
            "Diagnosis and Classification of Diabetes.* Diabetes Care (updated annually).\n"
            "2. World Health Organization. *Obesity and overweight* fact sheet: adult BMI classification.\n"
            "3. Whelton PK, et al. *2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, "
            "and Management of High Blood Pressure in Adults.* Hypertension. 2018;71:e13–e115.\n"
            "4. Diabetes Prevention Program Research Group. *Reduction in the incidence of type 2 "
            "diabetes with lifestyle intervention or metformin.* N Engl J Med. 2002;346:393–403.\n"
            "5. Smith JW, et al. *Using the ADAP learning algorithm to forecast the onset of diabetes "
            "mellitus.* Proc Annu Symp Comput Appl Med Care. 1988:261–265 (origin of the Pima dataset).\n"
            "6. US Centers for Disease Control and Prevention. *National Diabetes Prevention Program.*"
        )


# ===========================================================================
# 8. SIDEBAR AND PAGE ROUTING
# ===========================================================================
with st.sidebar:
    st.markdown("## 🩺 Diabetes Assessment")
    st.caption("Check your diabetes risk, understand your results and find practical next steps.")
    st.radio("Navigation", PAGES, key="page")
    st.divider()
    if st.session_state.result:
        r = st.session_state.result
        if r["prediction"] == 1:
            st.error("Assessment complete: likely diabetic")
        else:
            st.success("Assessment complete: not likely diabetic")
    else:
        st.info("No assessment completed yet")
    st.divider()
    st.caption("Educational tool only. Not a medical diagnosis. "
               "Consult a healthcare professional about diabetes concerns.")

try:
    model, scaler = load_artifacts()
except FileNotFoundError as err:
    st.error(
        f"Can't find `{Path(err.filename).name}`. Put `Diabetesmodel.pkl` and `scaler.pkl` "
        "in the same folder as this app, then refresh the page."
    )
    st.stop()

ranges = training_ranges(scaler)
page = st.session_state.page

if page == PAGES[0]:
    page_welcome()
elif page == PAGES[1]:
    page_assessment(model, scaler, ranges)
elif page == PAGES[2]:
    page_results()
elif page == PAGES[3]:
    page_prevention()
else:
    page_about(model, scaler, ranges)