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

# 📌 Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

# 📌 Function to load data with proper encoding
@st.cache_data
def load_data(scoreboard_path, matches_path, players_path):
    try:
        scoreboard_encoding = detect_encoding(scoreboard_path)
        matches_encoding = detect_encoding(matches_path)
        players_encoding = detect_encoding(players_path)

        scoreboard = pd.read_csv(scoreboard_path, encoding=scoreboard_encoding)
        matches = pd.read_csv(matches_path, encoding=matches_encoding)
        players = pd.read_csv(players_path, encoding=players_encoding)

        st.success("✅ CSV files loaded successfully!")
        return scoreboard, matches, players

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None, None, None  # Return None if there's an issue

# 📌 Set file paths
scoreboard_path = "Scoreboard.csv"
matches_path = "Matches.csv"
players_path = "Players.csv"

# 📌 Load Data
scoreboard_df, matches_df, players_df = load_data(scoreboard_path, matches_path, players_path)

if scoreboard_df is None or matches_df is None or players_df is None:
    st.stop()  # Stop execution if data failed to load

# ✅ Auto-detect Player Data Columns
st.write("🔍 Detected Player Data Columns:", list(players_df.columns))
players_df.columns = players_df.columns.str.strip()  # Remove extra spaces

# Rename columns if they exist
column_mapping = {
    "Player Name": "player",
    "Total Runs": "runs",
    "Wickets Taken": "wickets",
    "Catches Taken": "catches",
    "Run Outs": "run_outs",
    "Stumpings": "stumpings",
    "Balls Faced": "balls_faced",
    "Overs Bowled": "overs_bowled",
    "Economy Rate": "economy"
}

for old_col, new_col in column_mapping.items():
    if old_col in players_df.columns:
        players_df.rename(columns={old_col: new_col}, inplace=True)

# ✅ Fill Missing Columns with Default Values
required_columns = ["runs", "wickets", "catches", "run_outs", "stumpings", "balls_faced", "overs_bowled", "economy"]
for col in required_columns:
    if col not in players_df.columns:
        players_df[col] = 0  # Assign default value

# Convert data types
for col in required_columns:
    players_df[col] = pd.to_numeric(players_df[col], errors="coerce")

players_df.dropna(inplace=True)  # Remove missing values

# ✅ Fantasy Points System
def calculate_fantasy_points(df):
    """
    Calculate fantasy points for players based on Dream11 rules.
    """
    df["fantasy_points"] = 0

    # Batting Points
    df["fantasy_points"] += df["runs"] * 1  # 1 point per run
    df["fantasy_points"] += (df["runs"] >= 50) * 8  # Bonus for 50s
    df["fantasy_points"] += (df["runs"] >= 100) * 16  # Bonus for 100s

    if "balls_faced" in df.columns:  # Check if column exists before using it
        df["fantasy_points"] -= (df["balls_faced"] == 0) * 2  # -2 for ducks

    # Bowling Points
    df["fantasy_points"] += df["wickets"] * 25  # 25 points per wicket
    df["fantasy_points"] += (df["wickets"] >= 3) * 8  # 3-wicket haul bonus
    df["fantasy_points"] += (df["wickets"] >= 5) * 16  # 5-wicket haul bonus

    # Fielding Points
    df["fantasy_points"] += df["catches"] * 8  # 8 points per catch
    df["fantasy_points"] += df["run_outs"] * 6  # 6 points per run-out
    df["fantasy_points"] += df["stumpings"] * 12  # 12 points per stumping

    return df

players_df = calculate_fantasy_points(players_df)

# 📌 Player Leaderboard
st.subheader("🏅 Top 10 Players (Fantasy Points)")

expected_columns = ["player", "fantasy_points"]
if all(col in players_df.columns for col in expected_columns):
    leaderboard = players_df[["player", "fantasy_points"]].sort_values(by="fantasy_points", ascending=False).head(10)
    st.dataframe(leaderboard)
else:
    st.error("❌ Missing 'player' or 'fantasy_points' column in dataset!")
