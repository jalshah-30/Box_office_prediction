import streamlit as st
import pandas as pd
import joblib

# ============================
# Load Model
# ============================
model = joblib.load("best_lgbm_model.pkl")

st.set_page_config(
    page_title="Movie Box Office Prediction",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Box Office Collection Prediction")

st.write("Predict the expected Box Office Collection (Crores)")

st.divider()

# ============================
# User Inputs
# ============================

vote_average = st.slider(
    "IMDb Rating",
    1.0,
    10.0,
    7.0,
    0.1
)

vote_count = st.number_input(
    "Vote Count",
    min_value=0,
    value=10000
)

language = st.selectbox(
    "Language",
    [
        "hi",
        "ta",
        "te",
        "ml",
        "kn"
    ]
)

popularity = st.slider(
    "Popularity",
    0.0,
    100.0,
    50.0,
    0.1
)

genre = st.selectbox(
    "Genre",
    [
        "Action",
        "Drama",
        "Comedy",
        "Thriller",
        "Romance",
        "Crime",
        "Sci-Fi",
        "Adventure",
        "Family",
        "Mystery"
    ]
)

company = st.selectbox(
    "Production Company",
    [
        "Yash Raj Films",
        "Dharma Productions",
        "T-Series",
        "Mythri Movie Makers",
        "Hombale Films",
        "Lyca Productions",
        "Sun Pictures",
        "Excel Entertainment",
        "Red Chillies Entertainment",
        "AGS Entertainment"
    ]
)

actor = st.selectbox(
    "Lead Actor",
    [
        "Shah Rukh Khan",
        "Salman Khan",
        "Aamir Khan",
        "Ranbir Kapoor",
        "Ranveer Singh",
        "Deepika Padukone",
        "Alia Bhatt",
        "Prabhas",
        "Allu Arjun",
        "Yash",
        "Jr. NTR",
        "Ram Charan",
        "Vijay",
        "Ajith Kumar",
        "Suriya",
        "Akshay Kumar",
        "Hrithik Roshan",
        "Kartik Aaryan",
        "Kiara Advani",
        "Kareena Kapoor"
    ]
)

budget = st.number_input(
    "Budget (Crores)",
    min_value=1.0,
    value=100.0
)

director = st.selectbox(
    "Director",
    [
        "Rajkumar Hirani",
        "S. S. Rajamouli",
        "Lokesh Kanagaraj",
        "Mani Ratnam",
        "Atlee",
        "Zoya Akhtar",
        "Sanjay Leela Bhansali",
        "Anurag Kashyap",
        "Kabir Khan",
        "Rohit Shetty",
        "Karan Johar",
        "Ayan Mukerji",
        "Vetrimaaran",
        "Sukumar",
        "Trivikram Srinivas",
        "Nelson Dilipkumar",
        "T. J. Gnanavel",
        "Shankar",
        "Vikram Kumar",
        "Priyadarshan"
    ]
)

# ============================
# Predict
# ============================

if st.button("Predict Collection"):

    data = pd.DataFrame({

        "vote_average":[vote_average],

        "vote_count":[vote_count],

        "original_language":[language],

        "popularity":[popularity],

        "genres":[genre],

        "production_companies":[company],

        "cast":[actor],

        "budget_crore":[budget],

        "directors":[director]

    })

    prediction = model.predict(data)[0]

    st.success(
        f"Predicted Collection : ₹ {prediction:.2f} Crores"
    )

    roi = prediction / budget

    st.metric(
        "ROI",
        f"{roi:.2f}x"
    )

    if roi < 1:
        st.error("Expected Result : FLOP")

    elif roi < 2:
        st.warning("Expected Result : Average")

    elif roi < 4:
        st.info("Expected Result : HIT")

    else:
        st.success("Expected Result : BLOCKBUSTER")