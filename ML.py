import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
@st.cache
def load_data():
    try:
        scoreboard = pd.read_csv("Scoreboard.csv", encoding="utf-8")
        matches = pd.read_csv("Matches.csv", encoding="utf-8")
        players = pd.read_csv("Players.csv", encoding="utf-8")
        return scoreboard, matches, players
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

scoreboard_df, matches_df, players_df = load_data()

if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()

# Merge datasets
merged_df = pd.merge(scoreboard_df, matches_df[['match_no', 'venue', 'toss_winner']], on='match_no', how='left')
merged_df['toss_winner'] = merged_df['toss_winner'].fillna("Unknown")

# Label Encoding
all_teams = list(set(matches_df['toss_winner'].dropna().unique()))
all_venues = list(set(matches_df['venue'].dropna().unique()))

team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

venue_encoder = LabelEncoder()
venue_encoder.fit(all_venues)

# Apply Encoding to Data
merged_df['toss_winner'] = team_encoder.transform(merged_df['toss_winner'])
merged_df['venue'] = venue_encoder.transform(merged_df['venue'])

# Feature Engineering
merged_df['Run_Rate'] = merged_df['Home_team_run'] / (merged_df['Home_team_over'] + 1)
merged_df['Target_Score'] = merged_df['Away_team_run']

X = merged_df[['Home_team_wickets', 'Home_team_over', 'toss_winner', 'venue', 'Run_Rate', 'Target_Score']]
y_home = merged_df['Home_team_run']

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_home, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏏 IPL Score & Player Performance Predictor")

# Team Logos
team_logos = {
    "Mumbai Indians": "https://your_image_url/IM.jpeg",
    "Chennai Super Kings": "https://your_image_url/CSK.png",
    "Royal Challengers Bangalore": "https://your_image_url/RCB.jpeg",
    "Kolkata Knight Riders": "https://your_image_url/KKR.png"
}

t1 = st.selectbox("🏠 Select Batting Team", list(team_logos.keys()))
t2 = st.selectbox("🏹 Select Bowling Team", [team for team in team_logos.keys() if team != t1])

col1, col2 = st.columns(2)
for team, col in zip([t1, t2], [col1, col2]):
    with col:
        logo_path = team_logos.get(team, "default.png")
        st.image(logo_path, width=100)

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

    # Win Probability
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

# Player Performance Prediction
if players_df is not None:
    st.subheader("🔥 Player Performance Prediction")

    batting_team_players = players_df[players_df['team'] == t1]['player_name'].dropna().unique()
    bowling_team_players = players_df[players_df['team'] == t2]['player_name'].dropna().unique()
    
    all_players = np.concatenate((batting_team_players, bowling_team_players))
    player = st.selectbox("🔍 Select Player", all_players)

    player_stats = players_df[players_df['player_name'] == player]

    if not player_stats.empty and 'batting_avg' in player_stats.columns:
        avg_runs = player_stats['batting_avg'].values[0]
    else:
        avg_runs = 0

    if st.button("📈 Predict Player Runs"):
        form_factor = np.random.uniform(0.8, 1.2)
        predicted_runs = avg_runs * form_factor
        st.write(f"🏏 **Predicted Runs for {player}: {predicted_runs:.2f}**")
