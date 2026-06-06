
import streamlit as st
import requests

st.title("🔥 YouTube AI System (Production Version)")

views = st.number_input("Views", 50000)
likes = st.number_input("Likes", 5000)
comments = st.number_input("Comments", 500)

if st.button("Predict"):

    data = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_hours": 1000,
        "avg_view_duration_sec": 400,
        "ctr": 8,
        "retention_percent": 60,
        "video_length_sec": 900,
        "upload_hour": 18,
        "upload_day": 5
    }

    res = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data
    )

    st.write(res.json())
