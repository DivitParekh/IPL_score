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

# 📌 Detect File Encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

# 📌 Load Data
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
        st.error(f"❌ Error loading CSV: {e}")
        return None, None, None

# 📌 File Paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

# 📌 Load Data
scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)

if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()

# 📌 Merge Data
merged_df = pd.merge(scoreboard_df, matches_df[['match_no', 'venue', 'toss_winner']], on='match_no', how='left')
merged_df['toss_winner'].fillna("Unknown", inplace=True)

# 📌 Encode Categorical Data
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()

all_teams = list(set(matches_df['toss_winner'].dropna().unique()))
all_venues = list(set(matches_df['venue'].dropna().unique()))

team_encoder.fit(all_teams)
venue_encoder.fit(all_venues)

merged_df['toss_winner'] = team_encoder.transform(merged_df['toss_winner'])
merged_df['venue'] = venue_encoder.transform(merged_df['venue'])

# 📌 Feature Engineering
merged_df['Run_Rate'] = merged_df['Home_team_run'] / (merged_df['Home_team_over'] + 1)
merged_df['Target_Score'] = merged_df['Away_team_run']

# 📌 Features & Target Variable
X = merged_df[['Home_team_wickets', 'Home_team_over', 'toss_winner', 'venue', 'Run_Rate', 'Target_Score']]
y_home = merged_df['Home_team_run']

# 📌 Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_home, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 📌 UI
st.title("🏏 IPL Fantasy & Score Prediction")

# 📌 Fantasy Team Selection
st.sidebar.subheader("📝 Select Your Fantasy Team")
player_list = players_df["Player Name"].dropna().unique()
selected_players = st.sidebar.multiselect("🔹 Choose 11 Players", player_list, default=player_list[:11])
captain = st.sidebar.selectbox("👑 Select Captain (2x Points)", selected_players)
vice_captain = st.sidebar.selectbox("🏅 Select Vice-Captain (1.5x Points)", selected_players)

# 📌 Predict Fantasy Points
if st.sidebar.button("⚡ Calculate Fantasy Points"):
    if len(selected_players) != 11:
        st.sidebar.error("❌ You must select exactly 11 players!")
    else:
        # 📌 Assign Random Performance Data (For Testing)
        np.random.seed(42)
        player_stats = {
            "Player": selected_players,
            "Runs": np.random.randint(0, 100, len(selected_players)),
            "Fours": np.random.randint(0, 10, len(selected_players)),
            "Sixes": np.random.randint(0, 5, len(selected_players)),
            "Wickets": np.random.randint(0, 5, len(selected_players)),
            "Catches": np.random.randint(0, 3, len(selected_players)),
        }
        df_fantasy = pd.DataFrame(player_stats)

        # 📌 Calculate Fantasy Points
        df_fantasy["Points"] = (
            df_fantasy["Runs"] +
            df_fantasy["Fours"] * 2 +
            df_fantasy["Sixes"] * 3 +
            df_fantasy["Wickets"] * 25 +
            df_fantasy["Catches"] * 8
        )

        # 📌 Apply Captain & Vice-Captain Bonus
        df_fantasy.loc[df_fantasy["Player"] == captain, "Points"] *= 2
        df_fantasy.loc[df_fantasy["Player"] == vice_captain, "Points"] *= 1.5

        # 📌 Show Fantasy Leaderboard
        st.sidebar.subheader("🏆 Fantasy Points Leaderboard")
        st.sidebar.dataframe(df_fantasy.sort_values(by="Points", ascending=False))

# 📌 Match Score Prediction
st.subheader("📊 Match Score Prediction")

t1 = st.selectbox("🏠 Batting Team", all_teams)
t2 = st.selectbox("🏹 Bowling Team", [team for team in all_teams if team != t1])
toss_winner = st.selectbox("🎲 Toss Winner", [t1, t2])
wickets = st.number_input("🎯 Wickets Fallen", 0, 10)
overs = st.number_input("⏳ Overs Played", 0.0, 20.0, 0.1)
current_score = st.number_input("🏏 Current Score", 0, 500)
target_score = st.number_input("🎯 Target Score (If Chasing)", 0, 500)
venue = st.selectbox("📍 Venue", venue_encoder.classes_)
weather = st.selectbox("🌦️ Weather", ["Clear", "Cloudy", "Rainy", "Humid"])

# 📌 Encode Inputs
toss_encoded = team_encoder.transform([toss_winner])[0]
venue_encoded = venue_encoder.transform([venue])[0]
weather_factor = {"Clear": 1.0, "Cloudy": 0.9, "Rainy": 0.85, "Humid": 0.95}[weather]

# 📌 Predict Score
if st.button("⚡ Predict Score"):
    run_rate = current_score / (overs + 1)
    input_data = np.array([[wickets, overs, toss_encoded, venue_encoded, run_rate, target_score]])
    input_scaled = scaler.transform(input_data)
    predicted_score = model.predict(input_scaled)[0] * weather_factor
    st.success(f"🏆 Predicted Score: {predicted_score:.2f}")

    # 📌 Win Probability
    win_prob = (current_score / (predicted_score + 1)) * 100
    fig, ax = plt.subplots()
    ax.pie([win_prob, 100 - win_prob], labels=[t1, t2], autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax.set_title("Win Probability")
    st.pyplot(fig)
