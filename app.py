import streamlit as st
import pandas as pd
import joblib

model = joblib.load("youtube_pipeline.pkl")

st.title("🔥 YouTube AI Popularity Predictor")

views = st.number_input("Views", value=50000)
watch_time_hours = st.number_input("Watch Time Hours", value=1000)
avg_view_duration_sec = st.number_input("Avg View Duration (sec)", value=400)
likes = st.number_input("Likes", value=5000)
dislikes = st.number_input("Dislikes", value=100)
comments = st.number_input("Comments", value=500)
shares = st.number_input("Shares", value=1000)
subscribers_gained = st.number_input("Subscribers Gained", value=300)
impressions = st.number_input("Impressions", value=100000)
ctr = st.number_input("CTR", value=8.0)
retention_percent = st.number_input("Retention Percent", value=60.0)
video_length_sec = st.number_input("Video Length", value=900)
upload_hour = st.number_input("Upload Hour", value=18)
upload_day = st.number_input("Upload Day", value=5)

if st.button("Predict"):

    data = {}

    # Original features
    data["views"] = views
    data["watch_time_hours"] = watch_time_hours
    data["avg_view_duration_sec"] = avg_view_duration_sec
    data["likes"] = likes
    data["dislikes"] = dislikes
    data["comments"] = comments
    data["shares"] = shares
    data["subscribers_gained"] = subscribers_gained
    data["impressions"] = impressions
    data["ctr"] = ctr
    data["retention_percent"] = retention_percent
    data["video_length_sec"] = video_length_sec
    data["upload_hour"] = upload_hour
    data["upload_day"] = upload_day

    # Derived features
    eps = 1

    data["like_rate"] = likes / (views + eps)
    data["comment_rate"] = comments / (views + eps)
    data["share_rate"] = shares / (views + eps)
    data["dislike_rate"] = dislikes / (views + eps)
    data["subscriber_conversion_rate"] = subscribers_gained / (views + eps)

    data["watch_efficiency"] = watch_time_hours / (views + eps)
    data["retention_score"] = retention_percent
    data["impression_efficiency"] = views / (impressions + eps)

    data["total_engagement"] = likes + comments + shares

    data["virality_score"] = (
        likes + comments + shares
    ) / (views + eps)

    data["audience_quality_score"] = (
        retention_percent * ctr
    )

    data["engagement_score"] = (
        likes + comments + shares
    ) / (views + eps)

    # Normalized features
    data["n_views"] = views
    data["n_watch_time"] = watch_time_hours
    data["n_retention"] = retention_percent
    data["n_ctr"] = ctr
    data["n_engagement"] = data["engagement_score"]
    data["n_conversion"] = data["subscriber_conversion_rate"]
    data["n_virality"] = data["virality_score"]

    # NLP features (defaults)
    data["title_char_length"] = 20
    data["title_word_count"] = 4
    data["title_sentiment"] = 0
    data["title_subjectivity"] = 0
    data["title_quality_score"] = 50

    tfidf_cols = [
        'tfidf_advanced','tfidf_ai','tfidf_amazing','tfidf_analysis',
        'tfidf_best','tfidf_business','tfidf_challenge','tfidf_complete',
        'tfidf_course','tfidf_crypto','tfidf_cybersecurity','tfidf_data',
        'tfidf_easy','tfidf_education','tfidf_explained','tfidf_fast',
        'tfidf_fitness','tfidf_football','tfidf_future','tfidf_gaming',
        'tfidf_guide','tfidf_hidden','tfidf_insane','tfidf_learning',
        'tfidf_machine','tfidf_modern','tfidf_powerful',
        'tfidf_programming','tfidf_python','tfidf_review',
        'tfidf_robotics','tfidf_science','tfidf_secret',
        'tfidf_secrets','tfidf_smart','tfidf_technology',
        'tfidf_tips','tfidf_travel','tfidf_tricks',
        'tfidf_tutorial','tfidf_ultimate'
    ]

    for col in tfidf_cols:
        data[col] = 0

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    st.success(f"Predicted Popularity Score: {prediction:.2f}")
