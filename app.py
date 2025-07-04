import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# MUST be the first Streamlit command
st.set_page_config(page_title="🎬 Movie Revenue Predictor")

# Load model
model = joblib.load("movie_revenue_model.pkl")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[(df["revenue"] > 0) & (df["budget"] > 0)]
    df["genres"] = df["genres"].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')
    return df

df = load_data()

# App Title
st.title("🎥 Movie Revenue Predictor")
st.markdown("Enter movie details to predict its estimated **box office revenue**.")

# Input fields
budget = st.number_input("💸 Budget (USD)", min_value=10000, max_value=400_000_000, value=50000000)
runtime = st.slider("⏱ Runtime (minutes)", 60, 240, 120)
popularity = st.slider("🔥 Popularity Score", 0.0, 100.0, 20.0)
language = st.selectbox("🗣 Language", df["original_language"].unique())
genre = st.selectbox("🎭 Genre", df["genres"].unique())

# Predict revenue
if st.button("🔮 Predict Revenue"):
    input_df = pd.DataFrame({
        "budget": [budget],
        "runtime": [runtime],
        "popularity": [popularity],
        "original_language": [language],
        "genres": [genre],
    })

    revenue = model.predict(input_df)[0]
    st.success(f"🎉 Estimated Revenue: **${int(revenue):,}**")

    st.subheader("📋 Prediction Inputs Summary")
    st.write(input_df)

# Visualizations
st.markdown("---")
st.header("📊 Movie Dataset Insights")

# 1. Budget vs Revenue (Sample)
st.subheader("📈 Budget vs Revenue (Sample of 100 movies)")
sample = df.sample(100)
fig1, ax1 = plt.subplots()
sns.scatterplot(x=sample["budget"], y=sample["revenue"], ax=ax1)
ax1.set_xlabel("Budget")
ax1.set_ylabel("Revenue")
st.pyplot(fig1)

# 2. Revenue Distribution
st.subheader("💰 Revenue Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df["revenue"], bins=50, kde=True, ax=ax2)
ax2.set_xlim(0, df["revenue"].quantile(0.95))  # remove extreme outliers
st.pyplot(fig2)

# 3. Average Revenue by Genre
st.subheader("🎭 Average Revenue by Genre (Top 10)")
genre_avg = df.groupby("genres")["revenue"].mean().sort_values(ascending=False).head(10)
fig3, ax3 = plt.subplots()
sns.barplot(x=genre_avg.values, y=genre_avg.index, ax=ax3)
ax3.set_xlabel("Average Revenue")
st.pyplot(fig3)

# 4. Budget vs Revenue colored by Genre
st.subheader("🎨 Budget vs Revenue Colored by Genre")
sample2 = df.sample(200)
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=sample2, x="budget", y="revenue", hue="genres", ax=ax4)
ax4.set_xlim(0, sample2["budget"].max())
ax4.set_ylim(0, sample2["revenue"].max())
st.pyplot(fig4)

# 5. Runtime vs Revenue
st.subheader("⏱ Runtime vs Revenue")
fig5, ax5 = plt.subplots()
sns.scatterplot(data=df, x="runtime", y="revenue", ax=ax5)
ax5.set_ylim(0, df["revenue"].quantile(0.95))
st.pyplot(fig5)
