import streamlit as st
import pandas as pd
import joblib

model = joblib.load("youtube_pipeline.pkl")

st.title("🔥 YouTube AI Popularity Predictor")

views = st.number_input("Views", value=50000)
likes = st.number_input("Likes", value=5000)
comments = st.number_input("Comments", value=500)

if st.button("Predict"):

    data = pd.DataFrame([{
        "views": views,
        "likes": likes,
        "comments": comments
    }])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Popularity Score: {prediction:.2f}")
