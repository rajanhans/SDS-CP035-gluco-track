# app.py
# Streamlit app for Diabetes prediction (CDC 21-feature schema, no external .pkl).
# Collects user-friendly inputs, builds the exact 21-D numeric vector in training order,
# optionally applies per-feature standardization, and predicts 0/1 + probability.

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import os
import os

st.set_page_config(
    page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered"
)
st.title("🩺 Diabetes Risk Predictor (CDC 21 Features)")
st.write("Files in current directory:", os.listdir("."))


# ----------------------------
# Model definition (must match training architecture)
# ----------------------------
class TabularFFNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes,
        output_dim=1,
        dropout=0.0,
        use_batchnorm=True,
        activation="relu",
    ):
        super().__init__()
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "leaky_relu": nn.LeakyReLU(0.01)}
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")
        act = acts[activation]

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # single logit for BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


# ----------------------------
# Utilities
# ----------------------------
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_resource(show_spinner=False)
def load_model(
    pt_path_or_upl,
    fallback_input_dim=None,
    fallback_hidden=None,
    fallback_dropout=0.0,
    fallback_use_bn=True,
    fallback_activation="relu",
):
    """
    Accepts:
      - checkpoint dict with {"model_state", "input_dim", "hidden_sizes", ...}
      - plain state_dict
      - pickled nn.Module (TabularFFNN)
    Returns (model, meta)
    """
    if hasattr(pt_path_or_upl, "read"):  # uploaded file
        raw = torch.load(
            io.BytesIO(pt_path_or_upl.read()), map_location="cpu", weights_only=False
        )
    else:
        raw = torch.load(pt_path_or_upl, map_location="cpu", weights_only=False)

    meta = {}
    # Case 1: checkpoint dict
    if isinstance(raw, dict) and "model_state" in raw:
        cfg = raw.get("cfg", {})
        input_dim = raw.get("input_dim", fallback_input_dim)
        hidden_sizes = raw.get("hidden_sizes", fallback_hidden)
        output_dim = raw.get("output_dim", 1)
        if input_dim is None or hidden_sizes is None:
            raise ValueError(
                "Checkpoint missing input_dim/hidden_sizes. Provide them in the sidebar."
            )
        model = TabularFFNN(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=output_dim,
            dropout=cfg.get("dropout", fallback_dropout),
            use_batchnorm=cfg.get("use_batchnorm", fallback_use_bn),
            activation=cfg.get("activation", fallback_activation),
        )
        model.load_state_dict(raw["model_state"], strict=True)
        model.eval()
        meta.update(
            {
                "input_dim": input_dim,
                "hidden_sizes": hidden_sizes,
                "output_dim": output_dim,
            }
        )
        return model, meta

    # Case 2: entire model object
    if isinstance(raw, nn.Module):
        model = raw
        meta = {"input_dim": None, "hidden_sizes": None, "output_dim": 1}
        if isinstance(model, TabularFFNN):
            sizes = []
            for m in model.net:
                if isinstance(m, nn.Linear):
                    sizes.append((m.in_features, m.out_features))
            if sizes:
                meta["input_dim"] = sizes[0][0]
                meta["hidden_sizes"] = [out for _, out in sizes[:-1]]
                meta["output_dim"] = sizes[-1][1]
        model.eval()
        return model, meta

    # Case 3: plain state_dict
    if isinstance(raw, dict):
        if fallback_input_dim is None or fallback_hidden is None:
            raise ValueError(
                "State dict provided. Please specify input_dim and hidden_sizes in the sidebar."
            )
        model = TabularFFNN(
            input_dim=fallback_input_dim,
            hidden_sizes=fallback_hidden,
            output_dim=1,
            dropout=fallback_dropout,
            use_batchnorm=fallback_use_bn,
            activation=fallback_activation,
        )
        model.load_state_dict(raw, strict=True)
        model.eval()
        meta = {
            "input_dim": fallback_input_dim,
            "hidden_sizes": fallback_hidden,
            "output_dim": 1,
        }
        return model, meta

    raise ValueError("Unrecognized model file format.")


def predict_single(model, x_numeric: np.ndarray, threshold: float = 0.5):
    """x_numeric: shape (1, input_dim) float32; returns (label_int, prob_float, logit_float)."""
    with torch.no_grad():
        xt = torch.from_numpy(x_numeric.astype(np.float32))
        logit = float(model(xt).item())
        prob = float(sigmoid_np(logit))
        label = int(prob >= threshold)
    return label, prob, logit


# ----------------------------
# CDC 21-feature schema (exact names & order)
# ----------------------------
# Binary 0/1 (radios). NOTE: names match the dataset exactly.
BINARY_FEATURES = [
    ("HighBP", "High blood pressure diagnosed?"),
    ("HighChol", "High cholesterol diagnosed?"),
    ("CholCheck", "Had cholesterol check in past 5 years?"),
    ("Smoker", "Smoked ≥100 cigarettes in lifetime?"),
    ("Stroke", "Ever told you had a stroke?"),
    ("HeartDiseaseorAttack", "Coronary heart disease or myocardial infarction?"),
    ("PhysActivity", "Any physical activity in past 30 days (not job)?"),
    ("Fruits", "Consume fruit ≥1 time/day?"),
    ("Veggies", "Consume vegetables ≥1 time/day?"),
    ("HvyAlcoholConsump", "Heavy alcohol consumption?"),
    ("AnyHealthcare", "Any health care coverage?"),
    ("NoDocbcCost", "Couldn’t see doctor because of cost?"),
    ("DiffWalk", "Serious difficulty walking/climbing stairs?"),
]

# Numeric inputs (no SleepTime in this schema)
NUMERIC_FEATURES = [
    ("BMI", "Body Mass Index (BMI)", 10.0, 60.0, 0.5, 27.5),
    ("PhysHlth", "Physical health not good (days, last 30)", 0, 30, 1, 0),
    ("MentHlth", "Mental health not good (days, last 30)", 0, 30, 1, 0),
]

# GenHlth 1..5 (1=Excellent .. 5=Poor)
GENHLTH_OPTIONS = [
    ("Excellent (1)", 1),
    ("Very good (2)", 2),
    ("Good (3)", 3),
    ("Fair (4)", 4),
    ("Poor (5)", 5),
]

# Age ordinal 1..13 derived from age in years
AGECAT_BINS = [
    (18, 24),
    (25, 29),
    (30, 34),
    (35, 39),
    (40, 44),
    (45, 49),
    (50, 54),
    (55, 59),
    (60, 64),
    (65, 69),
    (70, 74),
    (75, 79),
    (80, 150),
]


def age_years_to_cat_ordinal(age_years: int) -> int:
    for i, (lo, hi) in enumerate(AGECAT_BINS, start=1):
        if lo <= age_years <= hi:
            return i
    return 13


# Education (1..6) and Income (1..8)
EDU_OPTIONS = [
    ("Never attended/only kindergarten (1)", 1),
    ("Grades 1-8 (2)", 2),
    ("Grades 9-11 (3)", 3),
    ("Grade 12 or GED (4)", 4),
    ("College 1-3 yrs (5)", 5),
    ("College 4+ yrs (6)", 6),
]
INCOME_OPTIONS = [
    ("<$10,000 (1)", 1),
    ("$10,000–$14,999 (2)", 2),
    ("$15,000–$19,999 (3)", 3),
    ("$20,000–$24,999 (4)", 4),
    ("$25,000–$34,999 (5)", 5),
    ("$35,000–$49,999 (6)", 6),
    ("$50,000–$74,999 (7)", 7),
    ("≥$75,000 (8)", 8),
]

# EXACT training order of the 21 features
FEATURE_ORDER = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

# Optional: per-feature standardization means/stds JSON
# Example: {"BMI":{"mean":28.1,"std":6.2},"MentHlth":{"mean":3.2,"std":6.7}, ...}
SCALER_JSON_DEFAULT = ""


# ----------------------------
# Sidebar: model, threshold, fallback arch, optional scaling
# ----------------------------
st.sidebar.header("⚙️ Model")
# Always resolve model path relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(script_dir, "best_model.pt")
model_path = st.sidebar.text_input(
    "Checkpoint (.pt/.pth) path", value=default_model_path
)
model_upl = st.sidebar.file_uploader("...or upload .pt/.pth", type=["pt", "pth"])
threshold = st.sidebar.slider(
    "Decision threshold (prob ≥ thr → 1)", 0.05, 0.95, 0.35, 0.01
)

st.sidebar.markdown("---")
st.sidebar.subheader("If checkpoint lacks architecture")
fallback_input_dim = st.sidebar.number_input(
    "input_dim (fallback)", min_value=1, value=21, step=1
)
fallback_hidden_str = st.sidebar.text_input("hidden_sizes (comma)", value="256,128,64")
fallback_hidden = [
    int(x.strip()) for x in fallback_hidden_str.split(",") if x.strip().isdigit()
]
fallback_dropout = st.sidebar.number_input(
    "dropout", min_value=0.0, max_value=0.9, value=0.0, step=0.05
)
fallback_use_bn = st.sidebar.checkbox("use_batchnorm", value=True)
fallback_activation = st.sidebar.selectbox(
    "activation", ["relu", "gelu", "leaky_relu"], index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional: scaling (mean/std)")
scaler_json_text = st.sidebar.text_area(
    "Per-feature mean/std JSON", value=SCALER_JSON_DEFAULT, height=120
)
SCALER = None
if scaler_json_text.strip():
    try:
        SCALER = json.loads(scaler_json_text)
        st.sidebar.success("Scaler JSON parsed ✓")
    except Exception as e:
        st.sidebar.error(f"Scaler JSON parse error: {e}")

# Prefer canonical sidecar files if the sidebar JSON was not provided
if SCALER is None:
    for p in [
        "scaler_params.json",
        "artifacts_aligned/scaler_params.json",
        "advanced/submissions/team-members/rajan-hans/scaler_params.json",
    ]:
        try:
            if Path(p).exists():
                SCALER = json.loads(open(p, "r").read())
                st.sidebar.info(f"Loaded scaler params from {p}")
                break
        except Exception:
            continue

# Load canonical feature order if present and use as FEATURE_ORDER
try:
    for p in [
        "feature_order.json",
        "artifacts_aligned/feature_order.json",
        "advanced/submissions/team-members/rajan-hans/feature_order.json",
    ]:
        if Path(p).exists():
            fo = json.loads(open(p, "r").read())
            if isinstance(fo, list) and len(fo) > 0:
                FEATURE_ORDER = fo
                st.sidebar.info(f"Loaded feature_order from {p}")
                break
except Exception:
    pass

# Load model (show errors instead of blank page)
try:
    model, meta = load_model(
        model_path if Path(model_path).exists() else model_upl,
        fallback_input_dim=fallback_input_dim,
        fallback_hidden=fallback_hidden,
        fallback_dropout=fallback_dropout,
        fallback_use_bn=fallback_use_bn,
        fallback_activation=fallback_activation,
    )
    input_dim = meta.get("input_dim", fallback_input_dim)
    st.success(
        f"Model loaded ✓  input_dim={input_dim}, hidden_sizes={meta.get('hidden_sizes', fallback_hidden)}"
    )
except Exception as e:
    st.error("Failed to load model:")
    st.code(str(e))
    st.stop()

# ----------------------------
# Friendly input form (radios/sliders/selects)
# ----------------------------
st.markdown("### Enter Patient Information")


# --- Binary radios in 2 columns, compact ---
colA, colB = st.columns(2)
raw = {}

for i, (key, label) in enumerate(BINARY_FEATURES):
    col = colA if i % 2 == 0 else colB
    choice = col.radio(
        label,
        options=[("No", 0), ("Yes", 1)],
        index=0,
        horizontal=True,
        key=f"bin_{key}",
    )
    raw[key] = choice[1]

# Sex (binary 0/1)

sex_choice = colA.radio(
    "Sex", options=[("Female", 0), ("Male", 1)], index=1, horizontal=True, key="bin_Sex"
)
raw["Sex"] = sex_choice[1]

# --- Group all numeric sliders together ---
with st.expander("Numeric Inputs", expanded=True):
    for key, label, lo, hi, step, default in NUMERIC_FEATURES:
        # Set custom defaults for PhysHlth and MentHlth
        if key == "PhysHlth" or key == "MentHlth":
            default_val = 15
        else:
            default_val = default
        if isinstance(step, float):
            raw[key] = st.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default_val),
                step=float(step),
                key=f"num_{key}",
            )
        else:
            raw[key] = st.slider(
                label,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default_val),
                step=int(step),
                key=f"num_{key}",
            )

# GenHlth (1..5)
gen_label, gen_val = st.selectbox(
    "General Health (1=Excellent .. 5=Poor)", GENHLTH_OPTIONS, index=2
)
raw["GenHlth"] = gen_val

# Age, Education, Income in columns as before
c1, c2, c3 = st.columns(3)
age_years = int(
    c1.number_input("Age (years)", min_value=18, max_value=120, value=50, step=1)
)
raw["Age"] = age_years_to_cat_ordinal(age_years)
edu_label, edu_val = c2.selectbox("Education level", EDU_OPTIONS, index=3)
raw["Education"] = edu_val
inc_label, inc_val = c3.selectbox("Household income", INCOME_OPTIONS, index=3)
raw["Income"] = inc_val


# ----------------------------
# Build numeric vector in EXACT training order (+ optional scaling)
# ----------------------------
def build_vector(raw_dict, feature_order, scaler=None):
    values = []
    names = []
    for feat in feature_order:
        if feat not in raw_dict:
            raise KeyError(f"Feature '{feat}' not found in collected inputs.")
        val = float(raw_dict[feat])
        if scaler and feat in scaler:
            mu = scaler[feat].get("mean", 0.0)
            sd = scaler[feat].get("std", 1.0) or 1.0
            val = (val - mu) / sd
        values.append(val)
        names.append(feat)
    vec = np.array(values, dtype=np.float32).reshape(1, -1)
    return vec, names, values


# Preview vector
with st.expander("🔎 Preview numeric vector (order must match training)"):
    try:
        vec, names, vals = build_vector(raw, FEATURE_ORDER, scaler=SCALER)
        df_preview = pd.DataFrame({"feature": names, "value": vals})
        st.dataframe(df_preview, width="stretch")
        st.caption(f"Vector length = {vec.shape[1]}  |  model input_dim = {input_dim}")
    except Exception as e:
        st.error(str(e))

# Predict
if st.button("Predict"):
    try:
        vec, names, vals = build_vector(raw, FEATURE_ORDER, scaler=SCALER)
        # st.write("**Debug: Model Input Vector**")
        # st.dataframe(pd.DataFrame({"feature": names, "value": vals}))
        # st.write(f"Shape: {vec.shape}, Input Dim: {input_dim}")
        if vec.shape[1] != input_dim:
            st.error(
                f"Vector length {vec.shape[1]} ≠ model input_dim {input_dim}. "
                f"Edit FEATURE_ORDER or provide the correct architecture in the sidebar."
            )
        else:
            label, prob, logit = predict_single(model, vec, threshold)
            # st.write(
            #     f"**Debug: Model Output** | Label: {label} | Probability: {prob:.4f} | Logit: {logit:.4f} | Threshold: {threshold}"
            # )
            outcome = "Diabetic (1)" if label == 1 else "Non-diabetic (0)"
            pred_text = f"Prediction: <b>{outcome}</b>  |  Probability: <b>{prob:.4f}</b>  (logit={logit:.4f}, thr={threshold:.2f})"
            if label == 1:
                st.markdown(
                    f"""
                    <div style='background-color:#ffcccc;padding:1em;border-radius:8px;font-weight:bold;font-size:1.1em;'>
                        {pred_text}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='background-color:#d4edda;padding:1em;border-radius:8px;font-weight:bold;font-size:1.1em;'>
                        {pred_text}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.exception(e)

st.markdown("---")
st.caption(
    "This app uses the CDC 21-feature schema. Ensure your trained model used the same feature order. "
    "If you standardized inputs during training, paste per-feature mean/std JSON in the sidebar."
)
