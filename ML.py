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
import sys

# Disable speech recognition on Streamlit Cloud
if "streamlit_cloud" in sys.modules:
    st.warning("🎤 Voice features are disabled in cloud deployment.")
else:
    import speech_recognition as sr
    from gtts import gTTS

# 📌 Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

# 📌 Function to load data
@st.cache_data
def load_data(scoreboard_path, matches_path, players_path):
    try:
        scoreboard_encoding = detect_encoding(scoreboard_path)
        matches_encoding = detect_encoding(matches_path)
        players_encoding = detect_encoding(players_path)

        scoreboard = pd.read_csv(scoreboard_path, encoding=scoreboard_encoding)
        matches = pd.read_csv(matches_path, encoding=matches_encoding)
        players = pd.read_csv(players_path, encoding=players_encoding)

        #st.success("✅ CSV files loaded successfully!")
        return scoreboard, matches, players

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None, None, None

# 📌 Set file paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

# 📌 Load Data
scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)

if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()  # Stop execution if data failed to load

# 📌 Debugging: Print column names
#st.write("🔍 Columns in Players.csv:", players_df.columns.tolist())

# Rename columns if incorrect
if "Player Name" in players_df.columns and "Total Runs" in players_df.columns:
    players_df.rename(columns={"Player Name": "player", "Total Runs": "runs"}, inplace=True)

# 📌 Function to generate speech from text (Only for local execution)
def speak_text(text):
    if "streamlit_cloud" not in sys.modules:
        tts = gTTS(text=text, lang="en")
        tts.save("response.mp3")
        os.system("start response.mp3")  # Windows
        # os.system("mpg321 response.mp3")  # Linux/Mac

# 📌 Streamlit UI - Voice Assistant
st.title("🎙️ IPL Voice Assistant & Score Predictor")

if st.button("🎤 Ask a Question"):
    query = st.text_input("🗣️ Type your question instead:")
    if query:
        # AI response based on queries
        if "top batsman" in query.lower():
            response = "Virat Kohli is the top run scorer in IPL history."
        elif "best team" in query.lower():
            response = "Mumbai Indians has won the most IPL titles."
        elif "who will win" in query.lower():
            response = "I can predict based on current stats. Click on Predict Score!"
        else:
            response = "I'm still learning! Ask me about IPL stats."

        st.info(f"🤖 AI: {response}")
        speak_text(response)

# 📌 IPL Prediction - Team Selection
st.subheader("🏏 IPL Score Prediction")

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

# 📌 Toss Winner Selection
toss_winner = st.selectbox("🎲 Toss Winner", [t1, t2])

# 📌 User Inputs for Prediction
wickets = st.number_input("🎯 Wickets Fallen", min_value=0, max_value=10)
overs = st.number_input("⏳ Overs Played", min_value=0.0, max_value=20.0, step=0.1)
current_score = st.number_input("🏏 Current Score", min_value=0, max_value=500)
target_score = st.number_input("🎯 Target Score (If Chasing)", min_value=0, max_value=500)
venue = st.selectbox("📍 Venue", matches_df['venue'].dropna().unique())
weather = st.selectbox("🌦️ Weather Conditions", ["Clear", "Cloudy", "Rainy", "Humid"])

# 📌 Weather Impact Factor
weather_factor = {"Clear": 1.0, "Cloudy": 0.9, "Rainy": 0.85, "Humid": 0.95}[weather]

# 📌 Predict Score Button
if st.button("⚡ Predict Score"):
    run_rate = current_score / (overs + 1)
    predicted_score = (current_score + (20 - overs) * run_rate) * weather_factor

    st.success(f"🏆 Predicted Score: {predicted_score:.2f}")

    # 📌 Win Probability Calculation
    if target_score > 0:
        required_run_rate = (target_score - current_score) / (20 - overs + 1)
        win_prob = min(max((run_rate / required_run_rate) * 100, 10), 90)
    else:
        win_prob = min(max((current_score / (predicted_score + 1)) * 100, 10), 90)

    # 📌 Display Win Probability
    fig, ax = plt.subplots()
    ax.pie([win_prob, 100 - win_prob], labels=[t1, t2], autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax.set_title("Win Probability Comparison")
    st.pyplot(fig)
    st.success(f"🏆 Win Probability: {win_prob:.2f}%")

# 📌 Leaderboard for Top Players


# 📌 Fix column names dynamically
players_df.columns = players_df.columns.str.strip().str.lower()  # Convert to lowercase for consistency
column_mapping = {"player_name": "player", "total_runs": "runs"}  # Define possible renaming
players_df.rename(columns={col: column_mapping[col] for col in players_df.columns if col in column_mapping}, inplace=True)

# 📌 Check if required columns exist
if "player" in players_df.columns and "runs" in players_df.columns:
    players_df["runs"] = pd.to_numeric(players_df["runs"], errors="coerce")  # Convert to numeric
    players_df.dropna(subset=["player", "runs"], inplace=True)

    leaderboard = (
        players_df.groupby("player")["runs"]
        .sum()
        .reset_index()
        .sort_values(by="runs", ascending=False)
        .head(10)
    )

    st.subheader("🏅 Top 10 Players by Runs")
    st.dataframe(leaderboard)
else:
    st.error(f"❌ Columns found: {players_df.columns.tolist()}")
    st.error("❌ Missing required columns for leaderboard!")
