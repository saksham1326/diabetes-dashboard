# ========================= app.py =========================
# Diabetes Readmission Dashboard — Gradient & Colorful (Theme 2: Neon Blue–Purple)
# Features: EDA (before/after/comparison), NLP (spaCy+NLTK), ML (6 models + ROC + Confusion + Comparison)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML / NLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# XGBoost (optional)
xgb_available = True
try:
    import xgboost as xgb
except Exception:
    xgb_available = False

# NLTK / spaCy (used on NLP tab)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import spacy

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(
    page_title="Diabetes Readmission — Gradient Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------- CUSTOM CSS (Neon Blue–Purple Gradient, soft cards, nice fonts) ----------
st.markdown("""
<style>
/* Gradient Background */
.stApp {
  background: linear-gradient(135deg, #0d1b2a 0%, #3a0ca3 50%, #4361ee 100%) fixed;
  color: #ffffff;
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5 { color: #f8f9ff; }

/* Cards / Containers */
.block-container { padding-top: 1.5rem; }
.card {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* KPI chips */
.kpi {
  background: linear-gradient(135deg, rgba(67,97,238,0.35), rgba(58,12,163,0.35));
  border: 1px solid rgba(255,255,255,0.18);
  padding: 16px 18px;
  border-radius: 16px;
  text-align: center;
  color: #ecf2ff;
}

/* Tables */
thead tr th { color: #ffffff !important; }
tbody tr td { color: #ffffff !important; }

/* Plot bg */
.plotly, .stPlotlyChart, .stAltairChart, .st-bf { background: transparent !important; }

/* Buttons */
.stButton>button {
  background: linear-gradient(135deg, #7209b7, #3a0ca3);
  color: #ffffff;
  border: 0;
  padding: 0.6rem 1rem;
  border-radius: 12px;
  transition: transform 0.05s ease-in-out;
}
.stButton>button:hover { transform: translateY(-2px); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(13,27,42,0.6);
  backdrop-filter: blur(8px);
  border-right: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>🩺 Diabetes Readmission — Gradient Dashboard</h1>", unsafe_allow_html=True)
st.caption("End-to-end EDA • NLP • ML with a modern, colorful UI (Neon Blue–Purple)")

# ---------- LOAD RAW DATA ----------
@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_csv("/content/diabetic_data.csv")
    return df

df_raw = load_raw()

# ---------- CLEAN DATA & FEATURE ENGINEERING ----------
def clean_and_engineer(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.replace("?", np.nan, inplace=True)

    # Drop rows with missing readmitted
    if "readmitted" in df.columns:
        df = df[df["readmitted"].notna()]

    # Fill numeric
    for col in df.select_dtypes(include=["int64","float64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Convert age like [70-80) -> midpoint 75
    def convert_age(a):
        a = str(a).replace("[","").replace(")","")
        low, high = a.split("-")
        return (int(low) + int(high)) / 2
    if "age" in df.columns:
        df["age_num"] = df["age"].apply(convert_age)

    return df

df_clean = clean_and_engineer(df_raw)

# Numeric columns for EDA/ML
numeric_cols = [
    c for c in [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "age_num"
    ] if c in df_clean.columns
]

# ---------- SIDEBAR NAV ----------
st.sidebar.header("🔎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "📝 NLP Profiling", "🤖 ML Models & Comparison"])

# ---------- KPI ROW ----------
def kpi_row(df):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='kpi'><h4>Total Records</h4><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)
    with c2:
        uniq = df["patient_nbr"].nunique() if "patient_nbr" in df.columns else len(df)
        st.markdown(f"<div class='kpi'><h4>Unique Patients</h4><h2>{uniq:,}</h2></div>", unsafe_allow_html=True)
    with c3:
        readm = df["readmitted"].value_counts().to_dict() if "readmitted" in df.columns else {}
        st.markdown(f"<div class='kpi'><h4>Readmission Labels</h4><h2>{readm}</h2></div>", unsafe_allow_html=True)
    with c4:
        meds = df["num_medications"].mean() if "num_medications" in df.columns else 0
        st.markdown(f"<div class='kpi'><h4>Avg. Medications</h4><h2>{meds:.1f}</h2></div>", unsafe_allow_html=True)

# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Project Overview")
    st.write("""
- **Goal:** Analyze and predict *hospital readmission* for diabetic patients.
- **Dataset:** UCI Diabetes Readmission (~100k rows, 50+ features)
- **This app provides:** clean *EDA* (before/after/comparison), *NLP profiling*, and *ML model comparison* with modern visuals.
    """)
    kpi_row(df_clean)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Sample Rows (Raw)")
    st.dataframe(df_raw.head(10))
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# EDA (BEFORE / AFTER / COMPARISON)
# =========================================================
elif page == "📊 EDA":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis — Clean, Readable, Focused")
    st.markdown("</div>", unsafe_allow_html=True)
    kpi_row(df_clean)

    tab1, tab2, tab3 = st.tabs(["🔵 Before Cleaning", "🟢 After Cleaning", "🟣 Comparison"])

    # BEFORE
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Missing Values (Before Cleaning)**")
        fig, ax = plt.subplots(figsize=(12,5))
        sns.heatmap(df_raw[numeric_cols].isna(), cbar=False, ax=ax)
        st.pyplot(fig)

        st.write("**Correlation Heatmap (Before Cleaning)**")
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(df_raw[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # AFTER
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Missing Values (After Cleaning)**")
        fig, ax = plt.subplots(figsize=(12,5))
        sns.heatmap(df_clean[numeric_cols].isna(), cbar=False, ax=ax)
        st.pyplot(fig)

        st.write("**Correlation Heatmap (After Cleaning)**")
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap="viridis", ax=ax)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # COMPARISON
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Total Missing Values — Before vs After**")
        miss_before = df_raw.isna().sum().sum()
        miss_after = df_clean.isna().sum().sum()
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=["Before", "After"], y=[miss_before, miss_after], ax=ax)
        st.pyplot(fig)

        st.write("**Time in Hospital — KDE Comparison**")
        fig, ax = plt.subplots(figsize=(12,5))
        sns.kdeplot(df_raw["time_in_hospital"], label="Before", fill=True)
        sns.kdeplot(df_clean["time_in_hospital"], label="After", fill=True)
        ax.legend()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# NLP PROFILING (spaCy + NLTK)
# =========================================================
elif page == "📝 NLP Profiling":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 NLP Profiling (NLTK + spaCy) — Quick Insights")

    # Download resources (silent)
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    # Build a synthetic note_text if not provided
    if "note_text" not in df_clean.columns:
        df_clean["note_text"] = (
            "Patient diagnosed with " + df_clean["diag_1"].astype(str) +
            " taking medication " + df_clean["metformin"].astype(str)
            if "diag_1" in df_clean.columns and "metformin" in df_clean.columns
            else "No clinical note available"
        )

    text = " ".join(df_clean["note_text"].astype(str))
    tokens = word_tokenize(text.lower())
    tokens_clean = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens_no_sw = [t for t in tokens_clean if t not in stop_words]
    freq = FreqDist(tokens_no_sw)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top 20 Frequent Words**")
        top_df = pd.DataFrame(freq.most_common(20), columns=["word","count"])
        st.dataframe(top_df, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x="word", y="count", data=top_df, ax=ax)
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

    # spaCy POS/NER over a sample
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        st.info("Installing spaCy model… run in a code cell: !python -m spacy download en_core_web_sm")
        nlp = None

    if nlp:
        sample_text = " ".join(df_clean["note_text"].astype(str).sample(min(300, len(df_clean))))
        doc = nlp(sample_text)
        pos_counts = {}
        for tok in doc:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
        st.write("**POS Tag Distribution (sample)**")
        st.dataframe(pd.DataFrame(pos_counts.items(), columns=["POS","Count"]).sort_values("Count", ascending=False))
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ML MODELS & COMPARISON (6 models)
# =========================================================
elif page == "🤖 ML Models & Comparison":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 Train & Compare Models (Binary: Readmitted vs Not)")
    st.caption("Target = 1 if readmitted (<30 or >30), else 0 (NO)")

    # Build target
    if "readmitted" not in df_clean.columns:
        st.error("Column 'readmitted' missing in dataset.")
        st.stop()
    df_ml = df_clean.copy()
    df_ml["readmitted_binary"] = df_ml["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

    # Feature matrix (numeric only, focused set)
    X = df_ml[numeric_cols].copy()
    y = df_ml["readmitted_binary"].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaled versions for LR / KNN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Build models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=400),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42),
    }
    if xgb_available:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist"
        )

    # Fit & evaluate
    results = {}
    preds = {}
    probas = {}

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_s)[:,1]
            elif hasattr(model, "decision_function"):
                from sklearn.preprocessing import MinMaxScaler
                dfc = model.decision_function(X_test_s).reshape(-1,1)
                y_proba = MinMaxScaler().fit_transform(dfc).ravel()
            else:
                y_proba = None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:,1]
            elif hasattr(model, "decision_function"):
                from sklearn.preprocessing import MinMaxScaler
                dfc = model.decision_function(X_test).reshape(-1,1)
                y_proba = MinMaxScaler().fit_transform(dfc).ravel()
            else:
                y_proba = None

        preds[name] = y_pred
        results[name] = accuracy_score(y_test, y_pred)
        probas[name] = y_proba

    # KPI row for best model
    best_model = max(results, key=results.get)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='kpi'><h4>Best Model</h4><h2>{best_model}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi'><h4>Best Accuracy</h4><h2>{results[best_model]*100:.2f}%</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='kpi'><h4># Features</h4><h2>{X.shape[1]}</h2></div>", unsafe_allow_html=True)

    # Accuracy table
    st.write("### 📊 Accuracy Comparison")
    acc_df = pd.DataFrame(
        [(k, f"{v*100:.2f}%") for k,v in results.items()],
        columns=["Model", "Accuracy"]
    ).sort_values("Accuracy", ascending=False)
    st.dataframe(acc_df, use_container_width=True)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="magma", ax=ax)
    plt.xticks(rotation=30, ha="right"); plt.ylim(0,1); plt.ylabel("Accuracy")
    st.pyplot(fig)

    # Tabs: Confusion matrices, ROC curves, Reports
    tab_cm, tab_roc, tab_report = st.tabs(["🔍 Confusion Matrices", "📈 ROC Curves", "📄 Classification Reports"])

    with tab_cm:
        for name in models.keys():
            st.markdown(f"**{name}**")
            cm = confusion_matrix(y_test, preds[name])
            fig, ax = plt.subplots(figsize=(4.5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)

    with tab_roc:
        fig, ax = plt.subplots(figsize=(9,6))
        for name in models.keys():
            if probas[name] is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, probas[name])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        st.pyplot(fig)

    with tab_report:
        for name, model in models.items():
            st.markdown(f"**{name}**")
            rep = classification_report(y_test, preds[name], output_dict=False)
            st.code(rep)

    st.markdown("</div>", unsafe_allow_html=True)

# ========================= END app.py =========================
