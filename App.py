import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")

def predict_sentiment(comment):
    if not comment or not isinstance(comment, str) or not comment.strip():
        return "Error: input must be a non-empty string"
    return model.predict([comment])[0]

st.title("YouTube Comment Sentiment Classifier")
comment = st.text_input("Type a comment:")
if st.button("Predict"):
    st.write(f"Prediction: **{predict_sentiment(comment)}**")
