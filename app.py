import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("best_catboost_model.pkl")
df=pd.read_csv("Input_data.csv")

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Movie Box Office Collection Prediction",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🎬 Movie Box Office Collection Prediction")
st.write("Predict the expected box office collection of a movie based on its features.")

st.markdown("---")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Enter Movie Details")

vote_count = st.sidebar.number_input(
    "Vote Count",
    min_value=0,
    value=1000
)

popularity = st.sidebar.slider(
    "Popularity",
    0.0,
    100.0,
    25.0
)

budget = st.sidebar.number_input(
    "Budget (Crores ₹)",
    min_value=1,
    value=100
)

language = st.sidebar.selectbox(
    "Original Language",
    [
        "Hindi",
        "Tamil",
        "Telugu",
        "Malayalam",
        "Kannada",
        "Bengali",
        "English",
        "Marathi",
        "Punjabi",
        "Gujarati",
        "Oriya",
        "Tibetan",
        "Assamese",
        "Urdu"
    ]
)

genres = st.sidebar.selectbox(
    "Genre",
    ['Political Drama', 'Buddy Comedy', 'Coming-of-Age', 'Drama',
       'Quest', 'Action Epic', 'Docudrama', 'Epic', 'Psychological Drama',
       'Conspiracy Thriller', 'One-Person Army Action', 'Gangster',
       'Crime', 'Medical Drama', 'Comedy', 'Suspense Mystery', 'Gun Fu',
       'Romantic Comedy', 'Action', 'Biography', 'Adventure',
       'Body Horror', 'Psychological Thriller', 'Quirky Comedy',
       'True Crime', 'Feel-Good Romance', 'Spy', 'Heist', 'Period Drama',
       'Cop Drama', 'Romantic Epic', 'Road Trip', 'Tragedy',
       'Legal Thriller', 'Supernatural Horror', 'Parody',
       'Artificial Intelligence', 'Desert Adventure', 'Dark Romance',
       'Historical Epic', 'Political Thriller', 'Monster Horror',
       'Tragic Romance', 'Psychological Horror', 'Drug Crime',
       'Dark Comedy', 'Showbiz Drama', 'Caper', 'Satire', 'Superhero',
       'Time Travel', 'Legal Drama', 'Jungle Adventure', 'Slapstick',
       'Martial Arts', 'Romance', 'Anime', 'Disaster', 'Serial Killer',
       'Police Procedural', 'Teen Romance', 'Mystery',
       'History Documentary', 'Boxing', 'Fairy Tale', 'Horror',
       'Sports Documentary', 'Computer Animation', 'Animation',
       'Raunchy Comedy', 'Thriller', 'Fantasy', 'Buddy Cop',
       'Swashbuckler', 'Mockumentary', 'Cyber Thriller', 'Family',
       'Globetrotting Adventure', 'Kung Fu', 'Screwball Comedy',
       'Whodunnit', 'Extreme Sport', 'Dystopian Sci-Fi', 'Teen Comedy',
       'Erotic Thriller'
    ]
)

directors_list = sorted(df["directors"].dropna().unique().tolist())

director = st.sidebar.selectbox(
    "Director",
    directors_list
)

cast_list = sorted(df["cast"].dropna().unique().tolist())

cast = st.sidebar.selectbox(
    "Lead Actor",
    cast_list
)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Collection"):

    input_df = pd.DataFrame({
        "vote_count":[vote_count],
        "original_language":[language],
        "popularity":[popularity],
        "genres":[genres],
        "cast":[cast],
        "budget":[budget],
        "directors":[director]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Box Office Collection: ₹ {prediction:.2f} Crores")

    profit = prediction - budget

    if profit > 0:
        st.success(f"Estimated Profit: ₹ {profit:.2f} Crores")
    else:
        st.error(f"Estimated Loss: ₹ {abs(profit):.2f} Crores")

    roi = (prediction / budget) * 100

    st.metric(
        "Return on Investment",
        f"{roi:.1f}%"
    )

st.markdown("---")

st.info(
"""
### Features Used
- Vote Count
- Popularity
- Budget
- Original Language
- Genre
- Director
- Lead Actor
"""
)