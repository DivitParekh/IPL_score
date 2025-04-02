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
    """Detect the encoding of a given file."""
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)  # Read a sample of the file
        result = chardet.detect(raw_data)  # Detect encoding
        return result['encoding']

@st.cache_data
def load_data(scoreboard_path, matches_path, players_path):
    """Load data from CSV files with detected encoding."""
    try:
        scoreboard = pd.read_csv(scoreboard_path, encoding=detect_encoding(scoreboard_path))
        matches = pd.read_csv(matches_path, encoding=detect_encoding(matches_path))
        players = pd.read_csv(players_path, encoding=detect_encoding(players_path))
        return scoreboard, matches, players
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None, None

# Set file paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

# Load data
scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)
if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()

# Merge datasets
merged_df = pd.merge(scoreboard_df, matches_df[['match_no', 'venue', 'toss_winner']], on='match_no', how='left')

# Encode categorical features
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
team_encoder.fit(matches_df['toss_winner'].dropna().unique())
venue_encoder.fit(matches_df['venue'].dropna().unique())
merged_df['toss_winner'] = team_encoder.transform(merged_df['toss_winner'].fillna("Unknown"))
merged_df['venue'] = venue_encoder.transform(merged_df['venue'].fillna("Unknown"))

# Train Model
X = merged_df[['Home_team_wickets', 'Home_team_over', 'toss_winner', 'venue']]
y_home = merged_df['Home_team_run']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_home, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏏 IPL Score & Player Performance Predictor")
st.sidebar.header("Select Prediction Mode")
mode = st.sidebar.radio("Choose an option:", ["Team Score Prediction", "Player Performance Prediction", "Team Comparison Dashboard"])

if mode == "Team Score Prediction":
    # Team Selection
    teams = matches_df['toss_winner'].dropna().unique()
    t1 = st.selectbox("Select Batting Team", teams)
    t2 = st.selectbox("Select Bowling Team", [team for team in teams if team != t1])
    toss_winner = st.selectbox("Toss Winner", [t1, t2])
    wickets = st.slider("Wickets Fallen", 0, 10, 3)
    overs = st.slider("Overs Played", 0.0, 20.0, 10.0, step=0.1)
    venue = st.selectbox("Venue", venue_encoder.classes_)
    
    if st.button("Predict Score"):
        input_data = scaler.transform([[wickets, overs, team_encoder.transform([toss_winner])[0], venue_encoder.transform([venue])[0]]])
        predicted_score = model.predict(input_data)[0]
        st.success(f"Predicted Score: {predicted_score:.2f}")

elif mode == "Player Performance Prediction":
    # Player Selection
    player = st.selectbox("Select Player", players_df['player_name'].unique())
    matches_played = players_df.loc[players_df['player_name'] == player, 'matches_played'].values[0]
    avg_runs = players_df.loc[players_df['player_name'] == player, 'avg_runs'].values[0]
    st.write(f"Matches Played: {matches_played}, Avg Runs: {avg_runs}")
    
    if st.button("Predict Runs"):
        predicted_runs = avg_runs + np.random.randint(-10, 20)  # Basic prediction logic
        st.success(f"Predicted Runs: {predicted_runs:.2f}")

elif mode == "Team Comparison Dashboard":
    # Team Comparison
    team1 = st.selectbox("Select Team 1", matches_df['toss_winner'].dropna().unique())
    team2 = st.selectbox("Select Team 2", [team for team in teams if team != team1])
    team1_wins = matches_df[matches_df['winner'] == team1].shape[0]
    team2_wins = matches_df[matches_df['winner'] == team2].shape[0]
    st.write(f"{team1} Wins: {team1_wins}, {team2} Wins: {team2_wins}")
    
    # Win probability visualization
    fig, ax = plt.subplots()
    ax.pie([team1_wins, team2_wins], labels=[team1, team2], autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    st.pyplot(fig)
    
    st.success("Team Comparison Complete!")
