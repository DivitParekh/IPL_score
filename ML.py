import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Get the directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define CSV file paths
scoreboard_path = os.path.join(base_dir, "Scoreboard.csv")
matches_path = os.path.join(base_dir, "Matches.csv")
players_path = os.path.join(base_dir, "Players.csv")

# Try loading files with error handling
@st.cache_data
def load_data():
    try:
        # Attempt to load the CSV files with a fallback encoding if UTF-8 fails
        scoreboard = pd.read_csv(scoreboard_path, encoding="utf-8")
        matches = pd.read_csv(matches_path, encoding="utf-8")
        players = pd.read_csv(players_path, encoding="utf-8")
        st.write("CSV files loaded successfully!")
        return scoreboard, matches, players
    except UnicodeDecodeError as e:
        # If UTF-8 fails, try with 'latin1' encoding (often works as a fallback)
        st.warning(f"UTF-8 decoding failed, trying 'latin1' encoding: {e}")
        scoreboard = pd.read_csv(scoreboard_path, encoding="latin1")
        matches = pd.read_csv(matches_path, encoding="latin1")
        players = pd.read_csv(players_path, encoding="latin1")
        return scoreboard, matches, players
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None, None

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
    st.title("🏏 IPL Score & Player Performance Predictor")

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

else:
    st.error("Failed to load CSV data. Please check your file paths and encoding.")
