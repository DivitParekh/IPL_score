import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Safe File Loading Function
def load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

# Load datasets
scoreboard_df = load_csv(r"C:\Users\admin\OneDrive\Documents\Desktop\ML\Scoreboard.csv")
matches_df = load_csv(r"C:\Users\admin\OneDrive\Documents\Desktop\ML\Matches.csv")
players_df = load_csv(r"C:\Users\admin\OneDrive\Documents\Desktop\ML\Players.csv")

if scoreboard_df is None or matches_df is None:
    st.error("❌ Essential datasets missing! Check file paths.")
    st.stop()

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
    "Mumbai Indians": r"C:\Users\admin\OneDrive\Documents\Desktop\ML\IM.jpeg",
    "Chennai Super Kings": r"C:\Users\admin\OneDrive\Documents\Desktop\ML\CSK.png",
    "Royal Challengers Bangalore": r"C:\Users\admin\OneDrive\Documents\Desktop\ML\RCB.jpeg",
    "Kolkata Knight Riders": r"C:\Users\admin\OneDrive\Documents\Desktop\ML\KKR.png"
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

        # Time Graph for Player Performance
        st.subheader("📊 Player Performance Over Time")
        overs_list = list(range(1, 21))
        runs_list = np.cumsum(np.random.randint(2, 10, size=len(overs_list)))
        
        fig, ax = plt.subplots()
        ax.plot(overs_list, runs_list, marker='o', linestyle='-', color='green', label='Runs Scored')
        ax.set_xlabel("Overs")
        ax.set_ylabel("Runs")
        ax.set_title(f"Performance of {player} Over Time")
        ax.legend()
        st.pyplot(fig)
# Bowler Performance Prediction
# Bowler Performance Prediction
if players_df is not None:
    st.subheader("🔥 Bowler Performance Prediction")

    # Select Bowler from Bowling Team
    bowling_team_players = players_df[players_df['team'] == t2]['player_name'].dropna().unique()
    bowler = st.selectbox("🎯 Select Bowler", bowling_team_players)

    # Get Bowler Stats
    bowler_stats = players_df[players_df['player_name'] == bowler]

    if not bowler_stats.empty and 'bowling_avg' in bowler_stats.columns:
        avg_wickets = bowler_stats['bowling_avg'].values[0]
        economy_rate = bowler_stats['economy'].values[0] if 'economy' in bowler_stats.columns else 8.0
        total_wickets = bowler_stats['total_wickets'].values[0] if 'total_wickets' in bowler_stats.columns else 0
        matches_played = bowler_stats['matches'].values[0] if 'matches' in bowler_stats.columns else 1
    else:
        avg_wickets = 0
        economy_rate = 8  # Default economy rate if no data
        total_wickets = 0
        matches_played = 1



# Generate Overs List (1-20)
overs_list = np.arange(1, 21)  # Overs from 1 to 20


base_wickets = np.random.poisson(lam=1, size=len(overs_list))  # Poisson for realism
wickets_per_over = np.clip(base_wickets + np.random.normal(0, 0.5, size=len(overs_list)), 0, 3).astype(int)  # Add variation

# Introduce a bias: Death overs (16-20) have slightly higher wicket probability
for i in range(len(overs_list)):
    if 16 <= overs_list[i] <= 20:
        wickets_per_over[i] += np.random.choice([0, 1], p=[0.6, 0.4])  # 40% chance of an extra wicket

# Bar Chart for Wickets Per Over
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(overs_list, wickets_per_over, color='crimson', alpha=0.8)

# Labels and Title
ax.set_xticks(overs_list)
ax.set_xlabel("Overs")
ax.set_ylabel("Wickets Taken")
ax.set_title("📊 Bowler Wickets Per Over (Bar Chart)")
ax.grid(axis='y', linestyle="--", alpha=0.6)

# Show plot in Streamlit
st.pyplot(fig)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate Model Performance
y_pred = model.predict(X_test)

# Calculate Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Metrics in Streamlit UI
st.subheader("📊 Model Evaluation Metrics")
st.write(f"📈 **Mean Absolute Error (MAE):** {mae:.2f}")

st.write(f"📉 **R² Score:** {r2:.2f}")

# You can use these metrics to assess how well your model is performing
