# train_movie_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Clean data
df = df[(df["revenue"] > 0) & (df["budget"] > 0)]

# Extract main genre and production company
df["genres"] = df["genres"].apply(lambda x: eval(x)[0]['name'] if eval(x) else 'Unknown')

# Features and target
features = ["budget", "runtime", "popularity", "original_language", "genres"]
target = "revenue"

df = df.dropna(subset=features)

X = df[features]
y = df[target]

# Preprocessing
categorical = ["original_language", "genres"]
numerical = ["budget", "runtime", "popularity"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder='passthrough')

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "movie_revenue_model.pkl")

print("✅ Model trained and saved as movie_revenue_model.pkl")
