import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, classification_report)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScope — Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & TRAIN ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train(path):
    df = pd.read_csv(path)
    X  = df.drop("target", axis=1)
    y  = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "auc":      round(roc_auc_score(y_test, y_proba) * 100, 2),
        "cm":       confusion_matrix(y_test, y_pred),
        "report":   classification_report(y_test, y_pred, output_dict=True),
        "fi":       pd.Series(model.feature_importances_, index=X.columns) .sort_values(ascending=False),
    }
    return df, model, X.columns.tolist(), metrics

# ── SIDEBAR — FILE UPLOAD ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("🫀 CardioScope")
    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    st.markdown("---")
    st.caption("Random Forest · scikit-learn · Streamlit")

if uploaded is None:
    st.title("🫀 CardioScope — Heart Disease Predictor")
    st.info("👈 Upload **heart_statlog_cleveland_hungary_final.csv** from the sidebar to get started.")
    st.stop()

df, model, feature_cols, metrics = load_and_train(uploaded)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "🔍 Feature Importance",
    "📈 Data Insights",
    "🩺 Risk Predictor",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Model Performance")
    st.caption("Random Forest Classifier · 80/20 train-test split · random_state=42")

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",    f"{metrics['accuracy']}%")
    c2.metric("ROC-AUC",     f"{metrics['auc']}%")
    cr = metrics["report"]
    c3.metric("Precision (Disease)", f"{round(cr['1']['precision']*100,1)}%")
    c4.metric("Recall (Disease)",    f"{round(cr['1']['recall']*100,1)}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Confusion matrix heatmap
    with col1:
        st.subheader("Confusion Matrix")
        cm = metrics["cm"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Disease", "Heart Disease"],
            y=["No Disease", "Heart Disease"],
            text_auto=True,
            color_continuous_scale=[[0,"#1a1a2e"],[0.5,"#4a4a8e"],[1,"#e05c5c"]],
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8edf5",
            height=340,
            coloraxis_showscale=False,
        )
        fig_cm.update_traces(textfont_size=20)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class metrics radar
    with col2:
        st.subheader("Per-Class Metrics")
        categories = ["Precision", "Recall", "F1-Score"]
        vals_0 = [cr["0"]["precision"], cr["0"]["recall"], cr["0"]["f1-score"]]
        vals_1 = [cr["1"]["precision"], cr["1"]["recall"], cr["1"]["f1-score"]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[v*100 for v in vals_0] + [vals_0[0]*100],
            theta=categories + [categories[0]],
            fill="toself", name="No Disease",
            line_color="#5ce0b8", fillcolor="rgba(92,224,184,0.15)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[v*100 for v in vals_1] + [vals_1[0]*100],
            theta=categories + [categories[0]],
            fill="toself", name="Heart Disease",
            line_color="#e05c5c", fillcolor="rgba(224,92,92,0.15)"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[85, 100],
                                tickfont=dict(color="#6b7a99")),
                angularaxis=dict(tickfont=dict(color="#e8edf5")),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8edf5",
            legend=dict(font=dict(color="#e8edf5")),
            height=340,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Classification report table
    st.subheader("Full Classification Report")
    report_df = pd.DataFrame({
        "Class":     ["No Disease (0)", "Heart Disease (1)", "Weighted Avg"],
        "Precision": [f"{cr['0']['precision']*100:.1f}%", f"{cr['1']['precision']*100:.1f}%", f"{cr['weighted avg']['precision']*100:.1f}%"],
        "Recall":    [f"{cr['0']['recall']*100:.1f}%",    f"{cr['1']['recall']*100:.1f}%",    f"{cr['weighted avg']['recall']*100:.1f}%"],
        "F1-Score":  [f"{cr['0']['f1-score']*100:.1f}%",  f"{cr['1']['f1-score']*100:.1f}%",  f"{cr['weighted avg']['f1-score']*100:.1f}%"],
        "Support":   [int(cr['0']['support']),             int(cr['1']['support']),             int(cr['weighted avg']['support'])],
    })
    st.dataframe(report_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Feature Importance")
    st.caption("Mean decrease in impurity across all 100 trees")

    fi = metrics["fi"]
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            x=fi.values * 100,
            y=fi.index,
            orientation="h",
            labels={"x": "Importance (%)", "y": "Feature"},
            color=fi.values,
            color_continuous_scale=["#5c8de0", "#5ce0b8", "#e05c5c"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8edf5",
            coloraxis_showscale=False,
            height=420,
            yaxis=dict(autorange="reversed"),
        )
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=fi.values,
            names=fi.index,
            hole=0.55,
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8edf5",
            legend=dict(font=dict(color="#e8edf5", size=11)),
            height=420,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Feature importance table
    st.subheader("Importance Scores")
    fi_df = pd.DataFrame({
        "Feature":    fi.index,
        "Importance": [f"{v*100:.2f}%" for v in fi.values],
        "Rank":       range(1, len(fi)+1),
    })
    st.dataframe(fi_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Dataset Insights")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",  len(df))
    c2.metric("Heart Disease",   int(df["target"].sum()))
    c3.metric("No Disease",      int((df["target"]==0).sum()))
    c4.metric("Features",        len(feature_cols))

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Age distribution
    with col1:
        st.subheader("Heart Disease by Age Group")
        age_bins = pd.cut(df["age"], bins=[20,30,40,50,60,70,80], labels=["20s","30s","40s","50s","60s","70s"])
        age_df = df.copy()
        age_df["Age Group"] = age_bins
        age_df["Status"] = age_df["target"].map({0:"No Disease", 1:"Heart Disease"})
        age_counts = age_df.groupby(["Age Group","Status"]).size().reset_index(name="Count")

        fig_age = px.bar(
            age_counts, x="Age Group", y="Count", color="Status", barmode="group",
            color_discrete_map={"No Disease":"#5ce0b8","Heart Disease":"#e05c5c"},
        )
        fig_age.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8edf5", legend=dict(font=dict(color="#e8edf5")), height=320,
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Target donut
    with col2:
        st.subheader("Target Distribution")
        counts = df["target"].value_counts()
        fig_donut = px.pie(
            values=counts.values,
            names=["No Disease","Heart Disease"],
            hole=0.6,
            color_discrete_sequence=["#5ce0b8","#e05c5c"],
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e8edf5",
            legend=dict(font=dict(color="#e8edf5")), height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Numerical distributions
    st.subheader("Numerical Feature Distributions")
    num_cols = ["age","resting bp s","cholesterol","max heart rate","oldpeak"]
    sel = st.selectbox("Select feature", num_cols, format_func=lambda x: x.title())
    df_plot = df.copy()
    df_plot["Status"] = df_plot["target"].map({0:"No Disease",1:"Heart Disease"})
    fig_box = px.box(
        df_plot, x="Status", y=sel, color="Status",
        color_discrete_map={"No Disease":"#5ce0b8","Heart Disease":"#e05c5c"},
        points="outliers",
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8edf5", showlegend=False, height=380,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e8edf5", height=480,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🩺 Real-Time Risk Predictor")
    st.caption("Inputs are fed directly into the trained Random Forest model.")
    st.warning("⚠️ For educational purposes only — not a medical diagnosis.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age     = st.number_input("Age",              min_value=20, max_value=80, value=52)
            sex     = st.selectbox("Sex",                 [("Male",1),("Female",0)], format_func=lambda x: x[0])
            cp      = st.selectbox("Chest Pain Type", [(1,"Typical Angina"),(2,"Atypical Angina"),
                                    (3,"Non-Anginal Pain"),(4,"Asymptomatic")], format_func=lambda x: x[1])
            rbp     = st.number_input("Resting BP (mmHg)", min_value=80, max_value=200, value=130)

        with c2:
            chol    = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])
            ecg     = st.selectbox("Resting ECG", [(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")], format_func=lambda x: x[1])

        with c3:
            mhr     = st.number_input("Max Heart Rate",    min_value=60, max_value=220, value=150)
            exang   = st.selectbox("Exercise-Induced Angina", [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])
            oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
            stslope = st.selectbox("ST Slope", [(1,"Upsloping"),(2,"Flat"),(3,"Downsloping")], format_func=lambda x: x[1])

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "age":                 age,
            "sex":                 sex[1],
            "chest pain type":     cp[0],
            "resting bp s":        rbp,
            "cholesterol":         chol,
            "fasting blood sugar": fbs[0],
            "resting ecg":         ecg[0],
            "max heart rate":      mhr,
            "exercise angina":     exang[0],
            "oldpeak":             oldpeak,
            "ST slope":            stslope[0],
        }])

        pred  = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]

        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            if pred == 1:
                st.error(f"### 🔴 Heart Disease Detected\nProbability: **{proba*100:.1f}%**")
            else:
                st.success(f"### 🟢 No Heart Disease\nProbability: **{(1-proba)*100:.1f}%**")

        with col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                title={"text": "Risk Probability (%)", "font": {"color": "#e8edf5"}},
                number={"suffix": "%", "font": {"color": "#e8edf5", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b7a99"},
                    "bar":  {"color": "#e05c5c" if pred == 1 else "#5ce0b8"},
                    "steps": [
                        {"range": [0,  40], "color": "rgba(92,224,184,0.15)"},
                        {"range": [40, 70], "color": "rgba(224,200,92,0.15)"},
                        {"range": [70,100], "color": "rgba(224,92,92,0.15)"},
                    ],
                    "threshold": {"line": {"color": "white","width": 2}, "value": 50},
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e8edf5", height=260,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)