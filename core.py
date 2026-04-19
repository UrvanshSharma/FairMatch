import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# ---------------- LOAD DATA ---------------- #
team_df = pd.read_csv("team_stats.csv")
player_df = pd.read_csv("player_stats.csv")

team_df = team_df.drop(columns=['Unnamed: 0'])
player_df = player_df.drop(columns=['Unnamed: 0'])

team_df = team_df.rename(columns={"name": "team"})
player_df = player_df.rename(columns={"name": "player"})

team_df = team_df[['team','rating','kd','kd_diff','total_maps']]
player_df = player_df[['player','teams','rating','kd','kd_diff']]

# ---------------- FEATURE ENGINEERING ---------------- #
team_df['consistency'] = 1 / (1 + abs(team_df['kd_diff']))
team_df['kd_efficiency'] = team_df['kd'] / team_df['rating']
team_df['experience'] = team_df['total_maps'] / team_df['total_maps'].max()

scaler = MinMaxScaler()
cols = ['consistency', 'kd_efficiency', 'experience']
team_df[cols] = scaler.fit_transform(team_df[cols])

team_df['tactical_intelligence'] = (
    0.4 * team_df['consistency'] +
    0.3 * team_df['kd_efficiency'] +
    0.3 * team_df['experience']
)

team_df['team_strength'] = (
    0.5 * team_df['rating'] +
    0.3 * team_df['kd'] +
    0.2 * team_df['tactical_intelligence']
)

# ---------------- TRAIN MODEL ---------------- #
def train_model():
    matches = []

    for i in range(len(team_df)):
        for j in range(i+1, len(team_df)):
            t1 = team_df.iloc[i]
            t2 = team_df.iloc[j]

            strength_diff = t1['team_strength'] - t2['team_strength']
            ti_diff = t1['tactical_intelligence'] - t2['tactical_intelligence']

            score = 1.5 * strength_diff + 3 * ti_diff
            prob = 1 / (1 + np.exp(-score))

            win = 1 if np.random.rand() < prob else 0

            matches.append([strength_diff, ti_diff, win])
            matches.append([-strength_diff, -ti_diff, 1 - win])

    match_df = pd.DataFrame(matches, columns=['strength_diff', 'ti_diff', 'win'])

    X = match_df[['strength_diff', 'ti_diff']]
    y = match_df['win']

    model = LogisticRegression()
    model.fit(X, y)

    return model

model = train_model()

# ---------------- FUNCTIONS ---------------- #
def predict_win(team1, team2):
    t1 = team_df[team_df['team'] == team1].iloc[0]
    t2 = team_df[team_df['team'] == team2].iloc[0]

    df_input = pd.DataFrame([{
        'strength_diff': t1['team_strength'] - t2['team_strength'],
        'ti_diff': t1['tactical_intelligence'] - t2['tactical_intelligence']
    }])

    return model.predict_proba(df_input)[0][1]


def find_best_match(team_name):
    best_team = None
    best_score = float('inf')

    for other in team_df['team']:
        if other == team_name:
            continue

        prob = predict_win(team_name, other)
        fairness = abs(prob - 0.5)

        if fairness < best_score:
            best_score = fairness
            best_team = other

    return best_team


def simulate_match(team1, team2, n=100):
    prob = predict_win(team1, team2)

    wins = sum(1 for _ in range(n) if random.random() < prob)

    return wins, n - wins, prob