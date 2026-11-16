"""
Machine Failure Risk Analysis (Streamlit)
------------------------------------------
Works with the files your notebook saves in its CURRENT folder (no artifacts/ folder):

    failure_model.pkl      anomaly_model.pkl     type_map.pkl     config.pkl
    cv_results.csv         test_predictions.csv  feature_importance.csv
    ai4i2020.csv           (optional - only for the Data Explorer page)

The app looks for these files in (1) the folder where app.py is, then
(2) the folder you ran `streamlit run` from. So just keep everything together.

Run:  streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

# ------------------------------------------------------------------ setup
st.set_page_config(page_title="Machine Failure Risk Analysis", page_icon="🛠️", layout="wide")
sns.set_theme(style="whitegrid")

# Folders searched for the notebook's output files (no hard-coded absolute paths)
SEARCH_DIRS = []
for _d in (Path(__file__).resolve().parent, Path.cwd()):
    if _d not in SEARCH_DIRS:
        SEARCH_DIRS.append(_d)

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLOR = {"Low": "#2e9e5b", "Medium": "#f0a202", "High": "#d64045"}
ACTIONS = {
    "Low": "Normal - no immediate action",
    "Medium": "Schedule inspection / monitor",
    "High": "Immediate maintenance required",
}
DEFAULT_TYPE_MAP = {"L": 0, "M": 1, "H": 2}      # same as the notebook
RENAME = {
    "Type": "type", "Air temperature [K]": "air_temp", "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
}
NUM_COLS = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
REQUIRED = ["failure_model.pkl", "anomaly_model.pkl", "config.pkl",
            "cv_results.csv", "test_predictions.csv", "feature_importance.csv"]


# ------------------------------------------------------------------ file finding
def find_file(name):
    """Return the first existing path for `name` in SEARCH_DIRS, else None."""
    for folder in SEARCH_DIRS:
        p = folder / name
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_models():
    model = joblib.load(find_file("failure_model.pkl"))
    anom = joblib.load(find_file("anomaly_model.pkl"))
    cfg = joblib.load(find_file("config.pkl"))
    tm_path = find_file("type_map.pkl")
    type_map = joblib.load(tm_path) if tm_path else DEFAULT_TYPE_MAP
    return model, anom, cfg, type_map


@st.cache_data
def load_tables():
    cv = pd.read_csv(find_file("cv_results.csv"), index_col=0)
    preds = pd.read_csv(find_file("test_predictions.csv"))
    imp = pd.read_csv(find_file("feature_importance.csv"))
    return cv, preds, imp


def clean_raw(df):
    """Same cleaning + features as the notebook (works for the CSV file or an upload)."""
    df = df.rename(columns=RENAME)
    df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], errors="ignore")
    df = df.dropna().drop_duplicates()
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power"] = df["torque"] * df["rpm"] * 2 * np.pi / 60
    df["wear_torque"] = df["tool_wear"] * df["torque"]
    return df


@st.cache_data
def load_raw():
    """Cleaned dataset for the EDA page. None if ai4i2020.csv is not found."""
    p = find_file("ai4i2020.csv")
    return clean_raw(pd.read_csv(p)) if p else None


# ------------------------------------------------------------------ helpers
def level_from_score(score, low_t, high_t):
    return "High" if score >= high_t else "Medium" if score >= low_t else "Low"


def add_engineered(df, type_map):
    df = df.copy()
    if df["type"].dtype == object:                      # L / M / H  ->  0 / 1 / 2
        df["type"] = df["type"].map(type_map)
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power"] = df["torque"] * df["rpm"] * 2 * np.pi / 60
    df["wear_torque"] = df["tool_wear"] * df["torque"]
    return df


def score_machines(df, model, anom, cfg, type_map):
    """Same scoring steps as the notebook, but for many rows at once."""
    df = add_engineered(df, type_map)
    prob = model.predict_proba(df[cfg["features"]])[:, 1]
    raw = -anom.score_samples(df[cfg["num_cols"]])
    anomaly = np.clip((raw - cfg["score_min"]) / (cfg["score_max"] - cfg["score_min"]), 0, 1)
    risk = cfg["w_fail"] * prob + cfg["w_anom"] * anomaly
    out = df.copy()
    out["failure_prob"] = prob.round(4)
    out["anomaly_score"] = anomaly.round(4)
    out["risk_score"] = risk.round(4)
    out["predicted_failure"] = (prob >= cfg["threshold"]).astype(int)
    out["risk_level"] = [level_from_score(r, cfg["low_t"], cfg["high_t"]) for r in risk]
    out["recommended_action"] = out["risk_level"].map(ACTIONS)
    return out


def risk_badge(level):
    color = RISK_COLOR[level]
    return (f"<div style='background:{color};color:white;padding:14px 18px;border-radius:10px;"
            f"font-size:1.4rem;font-weight:700;text-align:center'>{level.upper()} RISK</div>")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ load everything
missing = [f for f in REQUIRED if find_file(f) is None]
if missing:
    st.error("These files were not found: " + ", ".join(missing))
    st.info(
        "Run the whole notebook first (the last cells save the files in the notebook's folder). "
        "Then keep `app.py` in that same folder, or start the app from that folder.\n\n"
        "**Folders searched:**\n\n" + "\n".join(f"- `{d}`" for d in SEARCH_DIRS)
    )
    st.stop()

try:
    model, anom_model, cfg, TYPE_MAP = load_models()
    cv_results, test_preds, importance = load_tables()
except Exception as e:  # version mismatch / corrupted pickle
    st.error(f"Could not load the saved files: {e}")
    st.info("This usually means scikit-learn / xgboost versions differ from the notebook. "
            "Install the same versions, or re-run the notebook in this environment.")
    st.stop()

raw_df = load_raw()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛠️ Machine Failure Risk")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Explorer", "Model Performance", "Predict a Machine", "Maintenance Queue"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{cfg['best_name']}**")
st.sidebar.caption(f"Decision threshold: **{cfg['threshold']:.3f}**")
st.sidebar.caption("Dataset: AI4I 2020")

if cfg.get("sklearn_version") and cfg["sklearn_version"] != sklearn.__version__:
    st.sidebar.warning(
        f"Models were saved with scikit-learn {cfg['sklearn_version']}, you have {sklearn.__version__}. "
        "If you see errors, install the same version."
    )

# ================================================================== OVERVIEW
if page == "Overview":
    st.title("Statistical Anomaly Detection & Machine Failure Risk")
    st.write(
        "Predicts machine failure, detects abnormal operating conditions, and combines both into a "
        "**risk score** (Low / Medium / High) with a recommended maintenance action."
    )

    m = cfg["test_metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best model", cfg["best_name"])
    c2.metric("Recall", f"{m['recall']:.2f}")
    c3.metric("Precision", f"{m['precision']:.2f}")
    c4.metric("PR-AUC", f"{m['pr_auc']:.2f}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    st.caption("Test-set results. Failures are rare (~3.4%), so accuracy alone would be misleading.")

    st.subheader("Risk levels on the test set")
    table = (test_preds.groupby("risk_level")["actual_failure"]
             .agg(machines="count", failures="sum", failure_rate="mean")
             .reindex(RISK_ORDER).fillna(0))
    left, right = st.columns([1, 1])
    with left:
        show_tbl = table.copy()
        show_tbl["failure_rate"] = (show_tbl["failure_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(show_tbl)
        st.markdown(
            f"**Formula:** `risk = {cfg['w_fail']:.1f} × failure probability + "
            f"{cfg['w_anom']:.1f} × anomaly score`  \n"
            f"**Cut-offs:** Low < {cfg['low_t']:.2f} ≤ Medium < {cfg['high_t']:.2f} ≤ High"
        )
    with right:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.countplot(x="risk_level", hue="actual_failure", data=test_preds, order=RISK_ORDER, ax=ax)
        ax.set_title("Risk level vs actual failures")
        ax.set_xlabel("")
        ax.legend(title="Actual failure")
        show(fig)

    
# ================================================================== DATA EXPLORER
elif page == "Data Explorer":
    st.title("Data Explorer")
    if raw_df is None:
        st.warning("`ai4i2020.csv` was not found next to app.py. Upload it here to use this page.")
        up = st.file_uploader("Upload ai4i2020.csv", type="csv")
        if up is None:
            st.stop()
        raw_df = clean_raw(pd.read_csv(up))

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (cleaned)", f"{len(raw_df):,}")
    c2.metric("Failures", f"{int(raw_df['failure'].sum()):,}")
    c3.metric("Failure rate", f"{raw_df['failure'].mean():.2%}")

    tab1, tab2, tab3, tab4 = st.tabs(["Class balance", "Distributions", "Correlation", "Outliers"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        sns.countplot(x="failure", data=raw_df, ax=axes[0])
        axes[0].set_title("Failure vs no failure")
        sns.barplot(x="type", y="failure", data=raw_df, order=["L", "M", "H"], ax=axes[1])
        axes[1].set_title("Failure rate by machine type")
        plt.tight_layout()
        show(fig)

    with tab2:
        col = st.selectbox("Feature", NUM_COLS + ["temp_diff", "power", "wear_torque"])
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.histplot(data=raw_df, x=col, hue="failure", bins=40, kde=True, stat="density",
                     common_norm=False, ax=ax)
        ax.set_title(f"{col}: failed vs not failed")
        show(fig)
        st.caption(f"Skewness of {col}: {raw_df[col].skew():.2f}")
        st.dataframe(raw_df[NUM_COLS].describe().T.round(2))

    with tab3:
        feats = ["type"] + NUM_COLS + ["failure"]
        corr_df = raw_df.copy()
        corr_df["type"] = corr_df["type"].map(TYPE_MAP)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr_df[feats].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        show(fig)

    with tab4:
        z = (raw_df[NUM_COLS] - raw_df[NUM_COLS].mean()) / raw_df[NUM_COLS].std()
        z_flag = (z.abs() > 3).any(axis=1)
        q1, q3 = raw_df[NUM_COLS].quantile(0.25), raw_df[NUM_COLS].quantile(0.75)
        iqr = q3 - q1
        iqr_flag = ((raw_df[NUM_COLS] < q1 - 1.5 * iqr) | (raw_df[NUM_COLS] > q3 + 1.5 * iqr)).any(axis=1)

        a, b = st.columns(2)
        a.metric("Z-score flagged (|z| > 3)", int(z_flag.sum()))
        b.metric("IQR flagged", int(iqr_flag.sum()))
        st.write(
            f"Failure rate in IQR-flagged rows: **{raw_df.loc[iqr_flag, 'failure'].mean():.1%}** "
            f"vs others: **{raw_df.loc[~iqr_flag, 'failure'].mean():.1%}**"
        )
        top = raw_df.assign(z_score_max=z.abs().max(axis=1).round(3))[z_flag | iqr_flag]
        st.dataframe(top.sort_values("z_score_max", ascending=False).head(15))

# ================================================================== MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("Model Performance")

    st.subheader("Model comparison (5-fold cross-validation, train data)")
    st.dataframe(cv_results.style.highlight_max(axis=0, color="#cfeedd"))

    y_true = test_preds["actual_failure"].values
    prob = test_preds["failure_prob"].values
    pred = (prob >= cfg["threshold"]).astype(int)

    st.subheader(f"Test-set results: {cfg['best_name']}")
    c1, c2 = st.columns(2)
    with c1:
        cm = pd.crosstab(pd.Series(y_true, name="Actual"), pd.Series(pred, name="Predicted"))
        cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No fail", "Fail"], yticklabels=["No fail", "Fail"], ax=ax)
        ax.set_title("Confusion matrix")
        show(fig)
    with c2:
        st.dataframe(pd.Series(cfg["test_metrics"]).round(3).rename("value").to_frame())

    c3, c4 = st.columns(2)
    with c3:
        p, r, _ = precision_recall_curve(y_true, prob)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(r, p)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        show(fig)
    with c4:
        fpr, tpr, _ = roc_curve(y_true, prob)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        show(fig)

    st.subheader("permutation importance")
    imp = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_xlabel(" PR-AUC when shuffled")
    show(fig)

# ================================================================== PREDICT A MACHINE
elif page == "Predict a Machine":
    st.title("Predict a Machine")
    single, batch = st.tabs(["Single machine", "Batch (CSV upload)"])

    with single:
        st.write("Enter the machine readings to get the failure probability, anomaly score and risk level.")
        c1, c2 = st.columns(2)
        with c1:
            mtype = st.selectbox("Machine type (quality variant)", ["L", "M", "H"])
            air = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
            proc = st.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
        with c2:
            rpm = st.slider("Rotational speed [rpm]", 1100, 2900, 1500, 10)
            torque = st.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.5)
            wear = st.slider("Tool wear [min]", 0, 255, 100, 1)

        row = pd.DataFrame([{"type": mtype, "air_temp": air, "process_temp": proc,
                             "rpm": rpm, "torque": torque, "tool_wear": wear}])
        res = score_machines(row, model, anom_model, cfg, TYPE_MAP).iloc[0]
        level = res["risk_level"]

        st.markdown("---")
        a, b, c = st.columns(3)
        a.metric("Failure probability", f"{res['failure_prob']:.1%}")
        b.metric("Anomaly score (0-1)", f"{res['anomaly_score']:.2f}")
        c.metric("Risk score (0-1)", f"{res['risk_score']:.2f}")

        st.markdown(risk_badge(level), unsafe_allow_html=True)
        st.write("")
        st.progress(float(min(max(res["risk_score"], 0), 1)))
        st.info(f"**Recommended action:** {res['recommended_action']}")
        st.caption(
            f"Model predicts {'FAILURE' if res['predicted_failure'] == 1 else 'no failure'} "
            f"(probability {res['failure_prob']:.3f} vs threshold {cfg['threshold']:.3f})."
        )

        if raw_df is not None:
            notes = []
            for col, val in [("air_temp", air), ("process_temp", proc), ("rpm", rpm),
                             ("torque", torque), ("tool_wear", wear)]:
                q1, q3 = raw_df[col].quantile(0.25), raw_df[col].quantile(0.75)
                lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                if val < lo or val > hi:
                    notes.append(f"{col} = {val} is outside the usual range ({lo:.1f} to {hi:.1f})")
            if notes:
                st.warning("Unusual readings: " + "; ".join(notes))

    with batch:
        st.write(
            "Upload a CSV with columns `type, air_temp, process_temp, rpm, torque, tool_wear` "
            "(the original AI4I column names also work)."
        )
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            new = pd.read_csv(up).rename(columns=RENAME)
            missing_cols = [c for c in ["type"] + NUM_COLS if c not in new.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                scored = score_machines(new[["type"] + NUM_COLS], model, anom_model, cfg, TYPE_MAP)
                st.dataframe(scored.sort_values("risk_score", ascending=False))
                st.bar_chart(scored["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0))
                st.download_button("Download results", scored.to_csv(index=False).encode(),
                                   "scored_machines.csv", "text/csv")

# ================================================================== MAINTENANCE QUEUE
elif page == "Maintenance Queue":
    st.title("Maintenance Queue")
    st.write("Machines in the test set ranked by risk, riskiest first.")

    with st.expander("What-if: change the risk weights and cut-offs (exploratory)"):
        st.caption("Defaults are the notebook's choices. Changing these only re-labels machines here; "
                   "it doesn't retrain anything.")
        w_fail = st.slider("Weight on failure probability", 0.0, 1.0, float(cfg["w_fail"]), 0.05)
        low_t, high_t = st.slider("Low / High cut-offs", 0.0, 1.0,
                                  (float(cfg["low_t"]), float(cfg["high_t"])), 0.05)
    w_anom = 1 - w_fail

    q = test_preds.copy()
    q["risk_score"] = (w_fail * q["failure_prob"] + w_anom * q["anomaly_score"]).round(4)
    q["risk_level"] = [level_from_score(r, low_t, high_t) for r in q["risk_score"]]
    q["recommended_action"] = q["risk_level"].map(ACTIONS)

    counts = q["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Low", counts["Low"])
    c2.metric("🟠 Medium", counts["Medium"])
    c3.metric("🔴 High", counts["High"])

    f1, f2 = st.columns([2, 1])
    with f1:
        levels = st.multiselect("Show risk levels", RISK_ORDER, default=["High", "Medium"])
    with f2:
        top_n = st.number_input("Rows to show", 5, 500, 25, 5)

    show_cols = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
                 "failure_prob", "anomaly_score", "risk_score", "risk_level",
                 "recommended_action", "actual_failure"]
    view = q[q["risk_level"].isin(levels)].sort_values("risk_score", ascending=False)
    st.dataframe(view[show_cols].head(int(top_n)))
    st.download_button("Download full ranked list", q.sort_values("risk_score", ascending=False)
                       .to_csv(index=False).encode(), "maintenance_queue.csv", "text/csv")

    st.subheader("Actual failure rate per risk level")
    rate = (q.groupby("risk_level")["actual_failure"].mean().reindex(RISK_ORDER).fillna(0) * 100)
    st.bar_chart(rate)
    st.caption("A good risk score shows a clearly higher failure rate as the level goes from Low to High.")
