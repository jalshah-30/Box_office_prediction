# 🎬 AI-Powered Box Office Collection Prediction

An end-to-end Machine Learning project that predicts the **box office collection of Indian movies** using pre-release and audience engagement features. The project includes data preprocessing, model comparison, hyperparameter tuning, and an interactive **Streamlit dashboard** for real-time predictions.

---

## 📌 Project Overview

The Indian film industry releases hundreds of movies every year, making box office prediction a challenging task due to the influence of multiple factors such as budget, cast, director, genre, language, and audience engagement.

This project leverages Machine Learning to estimate a movie's expected box office collection based on historical movie data and presents the prediction through an intuitive Streamlit web application.

---

## 🚀 Features

- Predict estimated box office collection (₹ Crore)
- Estimate Profit/Loss
- Calculate Return on Investment (ROI)
- Commercial Risk Analysis
- Interactive Streamlit Dashboard
- Multiple Machine Learning models comparison
- Hyperparameter tuning
- Data preprocessing pipeline
- Model serialization for deployment

---

## 🗂 Dataset

The dataset contains approximately **1,200+ Indian movies** collected from publicly available sources.

### Features Used

- Budget (₹ Crore)
- Interested Count in Book My Show
- Original Language
- Primary Genre
- Director
- Lead Actor

### Target Variable

- Box Office Collection (₹ Crore)

---

## 🧹 Data Preprocessing

The following preprocessing techniques were applied:

- Missing value handling
- Categorical encoding
- One-Hot Encoding
- Feature engineering
- Data cleaning
- Train-Test Split
- Pipeline-based preprocessing

---

## 🤖 Machine Learning Models Evaluated

The following regression models were compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

---

## 🏆 Best Model

**Random Forest Regressor**

### Performance

| Metric | Score |
|---------|-------|
| R² Score | **0.698** |
| MAE | **33.14 Crore** |
| RMSE | **59.44 Crore** |

---

## 💻 Streamlit Dashboard

The application provides:

- Budget input
- Interested input
- Language selection
- Genre selection
- Director selection
- Lead Actor selection

### Outputs

- Estimated Gross Collection
- Profit / Loss
- Return on Investment (ROI)
- Commercial Risk Category

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib
- Streamlit
- Matplotlib
- Plotly

---

## 📁 Project Structure

```
BoxOfficePrediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Model/
│   ├──best_catboost_model.pkl
│
├── dataset/
│   └── movies.csv
│
├── notebooks/
│   └── EDA.ipynb
│
└── images/
    └── dashboard.png
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/box-office-prediction.git
```

Move into the project directory

```bash
cd box-office-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 Example Prediction

### Input

| Feature | Value |
|----------|-------|
| Budget | ₹125 Crore |
| Genre | Comedy |
| Director | Other |
| Lead Actor | Akshay Kumar |
| Language | Hindi |
| Vote Count | 21000 |
| Popularity | 50 |

### Output

| Metric | Prediction |
|----------|-----------|
| Estimated Collection | ₹164.25 Crore |
| Net Profit | ₹39.25 Crore |
| ROI | 131.4% |
| Risk | Moderate Risk (Profitable) |

---

## 📌 Future Improvements

- Include production company and writer information.
- Add trailer sentiment analysis.
- Incorporate release timing and holiday effects.
- Support multilingual movie prediction.
- Expand the dataset with additional movies.

---

## ⚠ Disclaimer

This application provides predictions based on a machine learning model trained on historical movie data. Actual box office performance depends on numerous unpredictable factors, including marketing, audience reception, competition, release timing, reviews, and external events. The predictions are intended for educational and demonstration purposes only.

---

## 👨‍💻 Author
**Jal Shah**

If you found this project useful, consider giving it a ⭐ on GitHub.