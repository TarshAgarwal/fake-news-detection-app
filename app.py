import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(layout="wide")
st.title("🕵️ Fake News Detection")

# Session state to store prediction history with timestamps
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.text_area("Enter News text:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # 🔍 Confidence score
        prob = model.predict_proba(input_vector)
        confidence = max(prob[0])
        st.write(f"🧠 Model confidence: `{confidence:.2f}`")

        # Store with timestamp if not duplicate
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not any(user_input == item[0] for item in st.session_state.history):
            st.session_state.history.append((user_input, prediction, timestamp))

        # Show result
        if prediction == 'FAKE':
            st.error("⚠️ Fake News detected!")
        else:
            st.success("✅ Legitimate News.")


# Display prediction history
st.markdown("---")
# 🧹 Clear button
if st.button("Clear History"):
    st.session_state.history.clear()
    st.success("Prediction history cleared!")
st.subheader("🧾 Prediction History")

# Two columns: FAKE on left, REAL on right
col1, col2 = st.columns(2)


with col1:
    st.markdown("### ❌ FAKE News")
    for text, label, timestamp in st.session_state.history:
        if label == 'FAKE':
            display = text if len(text) < 100 else text[:100] + "..."
            st.markdown(f"🔸 <span title='{text}'>{display}</span><br><small>🕓 {timestamp}</small>", unsafe_allow_html=True)

with col2:
    st.markdown("### ✅ REAL News")
    for text, label, timestamp in st.session_state.history:
        if label == 'REAL':
            display = text if len(text) < 100 else text[:100] + "..."
            st.markdown(f"🔹 <span title='{text}'>{display}</span><br><small>🕓 {timestamp}</small>", unsafe_allow_html=True)


st.markdown("---")
with st.expander("📘 View Test Dataset"):
    try:
        test_data = pd.read_csv("test_data.csv")
        st.write(test_data)
    except FileNotFoundError:
        st.error("Test dataset not found. Make sure 'test_data.csv' exists in the same folder as this app.")
