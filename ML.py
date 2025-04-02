import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import chardet

def detect_encoding(file_path):
    """Detect encoding to prevent UTF-8 errors."""
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

@st.cache_data
def load_data(scoreboard_path, matches_path, players_path):
    """Load datasets with encoding detection."""
    try:
        scoreboard = pd.read_csv(scoreboard_path, encoding=detect_encoding(scoreboard_path))
        matches = pd.read_csv(matches_path, encoding=detect_encoding(matches_path))
        players = pd.read_csv(players_path, encoding=detect_encoding(players_path))

        st.success("✅ CSV files loaded successfully!")
        return scoreboard, matches, players

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None, None, None

# File Paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

# Load Data
scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)

# Stop execution if data failed to load
if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()

# Merge Data
merged_df = pd.merge(scoreboard_df, matches_df[['match_no', 'venue', 'toss_winner']], on='match_no', how='left')
merged_df['toss_winner'].fillna("Unknown", inplace=True)

# Encoding Teams & Venues
team_encoder = LabelEncoder()
team_encoder.fit(matches_df['toss_winner'].dropna().unique())

venue_encoder = LabelEncoder()
venue_encoder.fit(matches_df['venue'].dropna().unique())

# Feature Engineering
merged_df['Run_Rate'] = merged_df['Home_team_run'] / (merged_df['Home_team_over'] + 1)
merged_df['Target_Score'] = merged_df['Away_team_run']

X = merged_df[['Home_team_wickets', 'Home_team_over', 'toss_winner', 'venue', 'Run_Rate', 'Target_Score']]
y_home = merged_df['Home_team_run']

# Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_home, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏏 IPL Score & Player Performance Predictor")

# Team Selection
team_logos = {
    "Mumbai Indians": "IM.jpeg",
    "Chennai Super Kings": "CSK.png",
    "Royal Challengers Bangalore": "RCB.jpeg",
    "Kolkata Knight Riders": "KKR.png"
}

t1 = st.selectbox("🏠 Select Batting Team", list(team_logos.keys()))
t2 = st.selectbox("🏹 Select Bowling Team", [team for team in team_logos.keys() if team != t1])

col1, col2 = st.columns(2)
for team, col in zip([t1, t2], [col1, col2]):
    with col:
        logo_path = team_logos.get(team, "default.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.warning(f"Logo not found for {team}")

# Toss Winner Selection
toss_winner = st.selectbox("🎲 Toss Winner", [t1, t2])

# User Inputs
wickets = st.number_input("🎯 Wickets Fallen", min_value=0, max_value=10)
overs = st.number_input("⏳ Overs Played", min_value=0.0, max_value=20.0, step=0.1)
current_score = st.number_input("🏏 Current Score", min_value=0, max_value=500)
target_score = st.number_input("🎯 Target Score (If Chasing)", min_value=0, max_value=500)
venue = st.selectbox("📍 Venue", venue_encoder.classes_)
weather = st.selectbox("🌦️ Weather Conditions", ["Clear", "Cloudy", "Rainy", "Humid"])

# Encode Inputs
toss_encoded = team_encoder.transform([toss_winner])[0]
venue_encoded = venue_encoder.transform([venue])[0]

# Weather Impact Factor
weather_factor = {"Clear": 1.0, "Cloudy": 0.9, "Rainy": 0.85, "Humid": 0.95}[weather]

# Predict Score
if st.button("⚡ Predict Score"):
    run_rate = current_score / (overs + 1)
    input_data = np.array([[wickets, overs, toss_encoded, venue_encoded, run_rate, target_score]])
    input_scaled = scaler.transform(input_data)
    predicted_score = model.predict(input_scaled)[0] * weather_factor
    
    st.success(f"🏆 Predicted Score: {predicted_score:.2f}")

    if target_score > 0:
        required_run_rate = (target_score - current_score) / (20 - overs + 1)
        win_prob = min(max((run_rate / required_run_rate) * 100, 10), 90)
    else:
        win_prob = min(max((current_score / (predicted_score + 1)) * 100, 10), 90)

    fig, ax = plt.subplots()
    ax.pie([win_prob, 100 - win_prob], labels=[t1, t2], autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax.set_title("Win Probability Comparison")
    st.pyplot(fig)
    st.success(f"🏆 Win Probability: {win_prob:.2f}%")

# 📊 **NEW FEATURES** 📊
st.header("🏅 Player & Team Insights")

# 1️⃣ **Top 10 Players by Runs**
if "player" in players_df.columns and "runs" in players_df.columns:
    leaderboard = (
        players_df.groupby("player")["runs"]
        .sum()
        .reset_index()
        .sort_values(by="runs", ascending=False)
        .head(10)
    )
    st.subheader("📌 Top 10 Players by Runs")
    st.dataframe(leaderboard)
else:
    st.error("❌ 'player' or 'runs' column missing in dataset!")

# 2️⃣ **Venue-Based Average Score Statistics**
if "venue" in matches_df.columns and "total_score" in scoreboard_df.columns:
    venue_stats = scoreboard_df.groupby("venue")["total_score"].mean().reset_index()
    st.subheader("📍 Venue-Based Average Score")
    fig, ax = plt.subplots()
    sns.barplot(data=venue_stats, x="venue", y="total_score", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# 3️⃣ **Team Performance Insights**
team_wins = matches_df['winning_team'].value_counts().reset_index()
team_wins.columns = ["Team", "Wins"]
st.subheader("🏆 Team Win Count")
st.dataframe(team_wins)

