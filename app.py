import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Preprocessor
# -----------------------------

model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Teen Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Teen Mental Health Prediction")

st.write(
    "Predict the likelihood of depression based on social media usage, sleep, stress, and lifestyle factors."
)

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        13,
        19,
        16
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    social_hours = st.slider(
        "Daily Social Media Hours",
        0.0,
        12.0,
        4.0
    )

    platform = st.selectbox(
        "Most Used Platform",
        [
            "Instagram",
            "Facebook",
            "TikTok",
            "Snapchat",
            "YouTube"
        ]
    )

    sleep = st.slider(
        "Sleep Hours",
        2.0,
        12.0,
        7.0
    )

with col2:

    screen_before_sleep = st.slider(
        "Screen Time Before Sleep (Hours)",
        0.0,
        6.0,
        2.0
    )

    academic = st.slider(
        "Academic Performance",
        0,
        100,
        70
    )

    physical = st.slider(
        "Physical Activity (Hours/Week)",
        0,
        20,
        5
    )

    interaction = st.selectbox(
        "Social Interaction Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    stress = st.slider(
        "Stress Level",
        1,
        10,
        5
    )

    anxiety = st.slider(
        "Anxiety Level",
        1,
        10,
        5
    )

    addiction = st.slider(
        "Addiction Level",
        1,
        10,
        5
    )

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_df = pd.DataFrame({

        "age":[age],

        "gender":[gender],

        "daily_social_media_hours":[social_hours],

        "platform_usage":[platform],

        "sleep_hours":[sleep],

        "screen_time_before_sleep":[screen_before_sleep],

        "academic_performance":[academic],

        "physical_activity":[physical],

        "social_interaction_level":[interaction],

        "stress_level":[stress],

        "anxiety_level":[anxiety],

        "addiction_level":[addiction]

    })

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(processed)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ High Risk of Depression")

    else:

        st.success("✅ Low Risk of Depression")

    st.metric(
        "Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))