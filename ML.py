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
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

@st.cache_data
def load_data(scoreboard_path, matches_path, players_path):
    try:
        scoreboard_encoding = detect_encoding(scoreboard_path)
        matches_encoding = detect_encoding(matches_path)
        players_encoding = detect_encoding(players_path)
        
        scoreboard = pd.read_csv(scoreboard_path, encoding=scoreboard_encoding)
        matches = pd.read_csv(matches_path, encoding=matches_encoding)
        players = pd.read_csv(players_path, encoding=players_encoding)
        
        return scoreboard, matches, players
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None, None

# File paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)

if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()

# Data Processing
merged_df = pd.merge(scoreboard_df, matches_df[['match_no', 'venue', 'toss_winner']], on='match_no', how='left')
merged_df['toss_winner'] = merged_df['toss_winner'].fillna("Unknown")

team_encoder = LabelEncoder()
teams = matches_df['toss_winner'].dropna().unique()
team_encoder.fit(teams)

venue_encoder = LabelEncoder()
venues = matches_df['venue'].dropna().unique()
venue_encoder.fit(venues)

merged_df['toss_winner'] = team_encoder.transform(merged_df['toss_winner'])
merged_df['venue'] = venue_encoder.transform(merged_df['venue'])

# Feature Engineering
merged_df['Run_Rate'] = merged_df['Home_team_run'] / (merged_df['Home_team_over'] + 1)
merged_df['Target_Score'] = merged_df['Away_team_run']

X = merged_df[['Home_team_wickets', 'Home_team_over', 'toss_winner', 'venue', 'Run_Rate', 'Target_Score']]
y_home = merged_df['Home_team_run']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_home, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI Enhancements
st.title("🏏 IPL Match & Player Performance Predictor")
st.sidebar.header("📊 Match Analysis Dashboard")

team_logos = {
    "Mumbai Indians": "IM.jpeg",
    "Chennai Super Kings": "CSK.png",
    "Royal Challengers Bangalore": "RCB.jpeg",
    "Kolkata Knight Riders": "KKR.png"
}

t1 = st.sidebar.selectbox("🏠 Select Batting Team", list(team_logos.keys()))
t2 = st.sidebar.selectbox("🏹 Select Bowling Team", [team for team in team_logos.keys() if team != t1])

toss_winner = st.sidebar.selectbox("🎲 Toss Winner", [t1, t2])
wickets = st.sidebar.number_input("🎯 Wickets Fallen", min_value=0, max_value=10)
overs = st.sidebar.number_input("⏳ Overs Played", min_value=0.0, max_value=20.0, step=0.1)
current_score = st.sidebar.number_input("🏏 Current Score", min_value=0, max_value=500)
target_score = st.sidebar.number_input("🎯 Target Score (If Chasing)", min_value=0, max_value=500)
venue = st.sidebar.selectbox("📍 Venue", venue_encoder.classes_)
weather = st.sidebar.selectbox("🌦️ Weather Conditions", ["Clear", "Cloudy", "Rainy", "Humid"])

# Team Stats
st.subheader("📊 Team Statistics")
team_stats = matches_df.groupby("toss_winner")["match_no"].count().reset_index()
team_stats.columns = ["Team", "Matches Played"]
st.dataframe(team_stats)

# Predict Score
if st.sidebar.button("⚡ Predict Score"):
    run_rate = current_score / (overs + 1)
    input_data = np.array([[wickets, overs, team_encoder.transform([toss_winner])[0], venue_encoder.transform([venue])[0], run_rate, target_score]])
    input_scaled = scaler.transform(input_data)
    predicted_score = model.predict(input_scaled)[0]
    
    st.success(f"🏆 Predicted Score: {predicted_score:.2f}")
    
    win_prob = np.clip((run_rate / (target_score / (20 - overs + 1))) * 100, 10, 90) if target_score > 0 else np.clip((current_score / (predicted_score + 1)) * 100, 10, 90)
    
    fig, ax = plt.subplots()
    ax.pie([win_prob, 100 - win_prob], labels=[t1, t2], autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax.set_title("Win Probability Comparison")
    st.pyplot(fig)
    
    st.success(f"🏆 Win Probability: {win_prob:.2f}%")
    
# Player Leaderboard
st.subheader("🏅 Player Leaderboard")
leaderboard = players_df.groupby("player")["runs"].sum().reset_index().sort_values(by="runs", ascending=False).head(10)
st.dataframe(leaderboard)
