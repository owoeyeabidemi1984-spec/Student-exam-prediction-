import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Exam Score Predictor", page_icon="🎓", layout="wide")

FEATURES = ["hours_studied", "attendance", "previous_score", "sleep_hours", "assignments_completed"]


def grade(score: float) -> str:
    if score >= 70:
        return "A - Excellent"
    if score >= 60:
        return "B - Good"
    if score >= 50:
        return "C - Average"
    return "F - Needs improvement"


@st.cache_data
def make_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "hours_studied": rng.integers(1, 11, n),
        "attendance": rng.integers(40, 101, n),
        "previous_score": rng.integers(30, 91, n),
        "sleep_hours": rng.integers(4, 10, n),
        "assignments_completed": rng.integers(0, 11, n),
    })

    def norm(col, lo, hi):
        return (df[col] - lo) / (hi - lo)

    df["final_score"] = (
        30 * norm("hours_studied", 1, 10)
        + 20 * norm("attendance", 40, 100)
        + 25 * norm("previous_score", 30, 90)
        + 10 * norm("sleep_hours", 4, 9)
        + 15 * norm("assignments_completed", 0, 10)
        + rng.normal(0, 3, n)
    ).clip(0, 100)
    return df


@st.cache_resource
def train(df: pd.DataFrame):
    X, y = df[FEATURES], df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

    lr = LinearRegression().fit(X_train_s, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_s, y_train)

    preds = {"Linear Regression": lr.predict(X_test_s), "Random Forest": rf.predict(X_test_s)}
    metrics = pd.DataFrame({
        name: {"R2": r2_score(y_test, p), "RMSE": float(np.sqrt(mean_squared_error(y_test, p)))}
        for name, p in preds.items()
    }).T
    return {"scaler": scaler, "models": {"Linear Regression": lr, "Random Forest": rf},
            "y_test": y_test, "preds": preds, "metrics": metrics}


def predict(bundle, model_name, values):
    row = pd.DataFrame([values], columns=FEATURES)
    scaled = pd.DataFrame(bundle["scaler"].transform(row), columns=FEATURES)
    return float(np.clip(bundle["models"][model_name].predict(scaled)[0], 0, 100))


df = make_dataset()
bundle = train(df)

st.title("🎓 Student Exam Score Predictor")
st.caption("Synthetic data of 1,000 students. Models: Linear Regression and Random Forest.")

# ---- Sidebar inputs ----
with st.sidebar:
    st.header("Student details")
    hours = st.slider("Hours studied per day", 1, 10, 6)
    attendance = st.slider("Attendance (%)", 40, 100, 85)
    prev = st.slider("Previous exam score", 30, 90, 70)
    sleep = st.slider("Sleep hours", 4, 9, 7)
    assignments = st.slider("Assignments completed (of 10)", 0, 10, 8)
    model_name = st.radio("Model", ["Random Forest", "Linear Regression"])

tab_pred, tab_eval, tab_data = st.tabs(["Prediction", "Model evaluation", "Dataset"])

with tab_pred:
    score = predict(bundle, model_name, [hours, attendance, prev, sleep, assignments])
    c1, c2 = st.columns(2)
    c1.metric("Predicted final score", f"{score:.1f} / 100")
    c2.metric("Grade", grade(score))
    st.progress(int(round(score)))

    other = "Linear Regression" if model_name == "Random Forest" else "Random Forest"
    st.caption(f"{other} predicts "
               f"{predict(bundle, other, [hours, attendance, prev, sleep, assignments]):.1f} for comparison.")

with tab_eval:
    st.subheader("Test-set metrics")
    st.dataframe(bundle["metrics"].style.format({"R2": "{:.3f}", "RMSE": "{:.2f}"}))
    st.info("The data follows a linear formula, so Linear Regression is expected to match "
            "or beat Random Forest here.")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(bundle["y_test"], bundle["preds"][model_name], alpha=0.6)
        ax.plot([0, 100], [0, 100], "r--", label="Perfect prediction")
        ax.set_xlabel("Actual score")
        ax.set_ylabel("Predicted score")
        ax.set_title(f"Actual vs Predicted ({model_name})")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        rf = bundle["models"]["Random Forest"]
        ax.bar(FEATURES, rf.feature_importances_)
        ax.set_title("Which factor matters most? (Random Forest)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

with tab_data:
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button("Download dataset (CSV)", df.to_csv(index=False),
                       "student_exam_dataset.csv", "text/csv")
    results = pd.DataFrame({"Actual": bundle["y_test"],
                            "Predicted_LR": bundle["preds"]["Linear Regression"],
                            "Predicted_RF": bundle["preds"]["Random Forest"]})
    st.download_button("Download predictions (CSV)", results.to_csv(index=False),
                       "predicted_scores.csv", "text/csv")
