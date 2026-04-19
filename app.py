from flask import Flask, request, jsonify
from flask_cors import CORS

# IMPORT CORE LOGIC
from core import team_df, predict_win, simulate_match, find_best_match

app = Flask(__name__)
CORS(app)

# ---------------- GET TEAMS ----------------
@app.route("/teams", methods=["GET"])
def get_teams():
    return jsonify(team_df['team'].tolist())


# ---------------- ANALYTICS (TOP GRAPHS) ----------------
@app.route("/analytics", methods=["GET"])
def analytics():
    return jsonify({
        "strength": team_df['team_strength'].tolist(),
        "tactical": team_df['tactical_intelligence'].tolist(),
        "teams": team_df['team'].tolist()
    })


# ---------------- MAIN PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    team1 = data.get("team1", "").strip()
    team2 = data.get("team2", "").strip()

    # SAFETY CHECK
    t1_df = team_df[team_df['team'] == team1]
    t2_df = team_df[team_df['team'] == team2]

    if t1_df.empty or t2_df.empty:
        return jsonify({"error": "Invalid team selection"}), 400

    t1 = t1_df.iloc[0]
    t2 = t2_df.iloc[0]

    prob = predict_win(team1, team2)
    wins1, wins2, _ = simulate_match(team1, team2)

    return jsonify({
        "team1": team1,
        "team2": team2,
        "prob1": prob,
        "prob2": 1 - prob,
        "wins1": wins1,
        "wins2": wins2,
        "metrics": ["Strength", "Tactical"],
        "team1_values": [
            t1['team_strength'],
            t1['tactical_intelligence']
        ],
        "team2_values": [
            t2['team_strength'],
            t2['tactical_intelligence']
        ]
    })


# ---------------- BEST MATCH COMPARISON ----------------
@app.route("/best-match-full", methods=["POST"])
def best_match_full():
    data = request.json

    team1 = data.get("team1", "").strip()
    team2 = data.get("team2", "").strip()

    # SAFETY CHECK
    if team1 not in team_df['team'].values or team2 not in team_df['team'].values:
        return jsonify({"error": "Invalid team"}), 400

    # ✅ GET DATAFRAME ROWS
    t1 = team_df[team_df['team'] == team1].iloc[0]
    t2 = team_df[team_df['team'] == team2].iloc[0]

    # ✅ FIND BEST MATCH
    best = find_best_match(team1)
    best_team = team_df[team_df['team'] == best].iloc[0]

    # CURRENT MATCH
    prob_r = predict_win(team1, team2)
    wins_r1, wins_r2, _ = simulate_match(team1, team2)

    # BEST MATCH
    prob_b = predict_win(team1, best)
    wins_b1, wins_b2, _ = simulate_match(team1, best)

    return jsonify({
        "random": {
            "opponent": team2,
            "prob": prob_r,
            "wins1": wins_r1,
            "wins2": wins_r2,
            "values": [
                t1['team_strength'],
                t1['tactical_intelligence']
            ],
            "opp_values": [
                t2['team_strength'],
                t2['tactical_intelligence']
            ]
        },
        "best": {
            "opponent": best,
            "prob": prob_b,
            "wins1": wins_b1,
            "wins2": wins_b2,
            "values": [
                t1['team_strength'],
                t1['tactical_intelligence']
            ],
            "opp_values": [
                best_team['team_strength'],
                best_team['tactical_intelligence']
            ]
        }
    })


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)