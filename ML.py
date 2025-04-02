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
    
    # Initialize empty dataframes to avoid unbound variable error
    scoreboard, matches, players = None, None, None  

    try:
        # Detect encoding
        scoreboard_encoding = detect_encoding(scoreboard_path)
        matches_encoding = detect_encoding(matches_path)
        players_encoding = detect_encoding(players_path)

        # Load CSV files with detected encoding
        scoreboard = pd.read_csv(scoreboard_path, encoding=scoreboard_encoding)
        matches = pd.read_csv(matches_path, encoding=matches_encoding)
        players = pd.read_csv(players_path, encoding=players_encoding)

       # st.write("✅ CSV files loaded successfully!")
    
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")

    return scoreboard, matches, players  # Return None if loading fails

# Set your file paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

# Load the data
scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)

# Check if data was loaded before proceeding
if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()  # Stop execution if data failed to load

# Load the data
scoreboard_df, matches_df, players_df = load_data()

# If the data is successfully loaded
if scoreboard_df is not None and matches_df is not None and players_df is not None:
    # Merge datasets
    merged_df = pd.merge(scoreboard_df, matches_df[['match_no', 'venue', 'toss_winner']], on='match_no', how='left')
    merged_df['toss_winner'] = merged_df['toss_winner'].fillna("Unknown")

    # Ensure LabelEncoders are trained with all possible values
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
    st.title("🏏 IPL Match & Player Performance Predictor")

    # Team Selection with Logos
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

    # Toss Winner Selection (Filtered Based on Teams)
    toss_winner = st.selectbox("🎲 Toss Winner", [t1, t2])

    # User Inputs
    wickets = st.number_input("🎯 Wickets Fallen", min_value=0, max_value=10)
    overs = st.number_input("⏳ Overs Played", min_value=0.0, max_value=20.0, step=0.1)
    current_score = st.number_input("🏏 Current Score", min_value=0, max_value=500)
    target_score = st.number_input("🎯 Target Score (If Chasing)", min_value=0, max_value=500)
    venue = st.selectbox("📍 Venue", venue_encoder.classes_)
    weather = st.selectbox("🌦️ Weather Conditions", ["Clear", "Cloudy", "Rainy", "Humid"])

    # Encode Inputs
    try:
        toss_encoded = team_encoder.transform([toss_winner])[0]
    except ValueError:
        st.error(f"❌ Toss winner '{toss_winner}' not recognized!")
        st.stop()

    try:
        venue_encoded = venue_encoder.transform([venue])[0]
    except ValueError:
        st.error(f"❌ Venue '{venue}' not recognized!")
        st.stop()

    # Weather Impact Factor
    weather_factor = {"Clear": 1.0, "Cloudy": 0.9, "Rainy": 0.85, "Humid": 0.95}[weather]

    # Predict Score
    if st.button("⚡ Predict Score"):
        run_rate = current_score / (overs + 1)
        input_data = np.array([[wickets, overs, toss_encoded, venue_encoded, run_rate, target_score]])
        input_scaled = scaler.transform(input_data)
        predicted_score = model.predict(input_scaled)[0] * weather_factor
        
        st.success(f"🏆 Predicted Score: {predicted_score:.2f}")
