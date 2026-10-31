import os
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==========================================================
AUTHOR_NAME = "Pratik Jena"          
AUTHOR_DETAILS = "BTech Final Year Project"   
# ==========================================================

st.set_page_config(
    page_title="Fake Review Detection - Prediction",
    page_icon="🤖",
    layout="wide",
)

# -------------------------------
# Sidebar: author info
# -------------------------------
with st.sidebar:
    st.markdown("### 👤 Developed by")
    st.markdown(f"**{AUTHOR_NAME}**")
    st.caption(AUTHOR_DETAILS)


# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model_vectorizer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model files not found. Please ensure 'model.pkl' and 'tfidf_vectorizer.pkl' exist in the same folder as app.py.")
        return None, None
    return joblib.load(model_path), joblib.load(vectorizer_path)


model, vectorizer = load_model_vectorizer()
if model is None or vectorizer is None:
    st.stop()

# -------------------------------
# Title
# -------------------------------
st.title("Fake Review Detection - Single Prediction")
st.caption(f"Developed by **{AUTHOR_NAME}** | {AUTHOR_DETAILS}")

# -------------------------------
# Single Prediction Section
# -------------------------------
review_text = st.text_area("Enter review text:", height=150)
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Predict"):
    if not review_text.strip():
        st.error("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            x_text = vectorizer.transform([review_text])
            dummy = np.zeros((1, 11))
            feats = np.hstack((x_text.toarray(), dummy))
            try:
                probabilities = model.predict_proba(feats)[0]
            except Exception:
                probabilities = None
                prediction = model.predict(feats)[0]
            else:
                prediction = 1 if probabilities[1] >= threshold else 0
            result = "Computer generated" if prediction == 0 else "Original"
            st.success(f"Prediction: {result}")
            if probabilities is not None:
                df_probs = pd.DataFrame({
                    "Label": ["Computer Generated", "Original"],
                    "Probability": [probabilities[0], probabilities[1]],
                })
                bar_chart = alt.Chart(df_probs).mark_bar(color="#66b3ff").encode(
                    x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                    y="Probability",
                ).properties(title="Prediction Probability Distribution", width=300, height=400)
                st.altair_chart(bar_chart, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(f"© {AUTHOR_NAME} - {AUTHOR_DETAILS}")
