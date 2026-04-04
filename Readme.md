# 🎮 Esports Matchmaking System using Machine Learning

## 📌 Project Description
This project builds a **data-driven esports matchmaking system** that creates balanced matches between teams using machine learning.

Instead of simple clustering, the system uses:
- Feature engineering (Tactical Intelligence)
- Match prediction (Logistic Regression)
- Simulation-based validation

The goal is to ensure **fair and competitive matches**.

---

## 🎯 Objectives
- Analyze team and player performance
- Engineer meaningful features (Tactical Intelligence)
- Predict match outcomes using ML
- Recommend fair opponents
- Compare random vs optimized matchmaking
- Validate fairness using simulation

---

## 📊 Dataset Description
The dataset includes:

### Team Data:
- **Rating** → Overall performance score  
- **KD (Kill/Death Ratio)** → Combat efficiency  
- **KD Difference** → Stability indicator  
- **Total Maps** → Experience  

### Player Data:
- Rating  
- KD  
- KD Difference  

---

## 🧠 Key Features

### 🔹 Tactical Intelligence (Engineered Feature)
A custom feature combining:
- Consistency (based on KD difference)
- Efficiency (KD vs rating)
- Experience (total matches)

---

### 🔹 Team Strength
Weighted score combining:
- Rating
- KD
- Tactical Intelligence

---

### 🔹 Match Prediction Model
- Logistic Regression
- Input: Difference between two teams
- Output: Win probability

---

### 🔹 Matchmaking Logic
- Finds opponent with **closest to 50% win probability**
- Ensures fair matches

---

### 🔹 Simulation
- Simulates 100 matches using probability
- Validates real-world outcomes

---

## ⚙️ Technologies Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Plotly (for interactive visualization)
- Scikit-learn

---

## 📈 Methodology

1. Data Cleaning
2. Feature Engineering
3. Team Strength Calculation
4. Match Pair Generation
5. Model Training (Logistic Regression)
6. Match Prediction
7. Matchmaking Optimization
8. Simulation & Visualization

---

## 📊 Results & Insights

- Random matchmaking often produces unbalanced matches  
- Optimized matchmaking produces near 50-50 outcomes  
- Tactical Intelligence improves prediction quality  
- Simulation confirms fairness of suggested matches  

---

## ⚖️ Evaluation Metric

### Fairness Score: 
| win_probability - 0.5 |

- Closer to 0 → more balanced match  
- Used to compare random vs optimized matchmaking  

---

## ⚠️ Errors Faced & Fixes

### 1: Flat Predictions (50-50 always)

## Cause:
 - Low variance in team strength

## Fix:
 - Adjusted model weighting: score = 1.5 * strength_diff + 3 * ti_diff

### 2. Poor Visualization
## Cause:
 - Using raw values instead of differences

## Fix:
Switched to:
 - Difference-based graphs
 - Clear comparison plots

### 3. Unbalanced "Best Match"
## Cause:
 - Matching based only on strength

## Fix:
 - Used fairness-based matchmaking: min|probability-0.5|

### Project Structure :
📁 esports-matchmaking
│
├── esports_matchmaking.ipynb
├── player_stats.csv
├── team_stats.csv
├── README.md
└── requirements.txt

### How to Run

## 1.Install Dependencies
    pip install -r requirements.txt

## 2.Run Notebook
    jupyter notebook esports_matchmaking.ipynb

### Key Insights
    We improve matchmaking fairness by minimizing win probability deviation instead of relying solely on raw performance metrics.