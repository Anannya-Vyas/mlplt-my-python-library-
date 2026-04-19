import sys
import os

# INJECT THE PYPI CODE AT THE START OF THE PATH
sys.path.insert(0, os.path.abspath("./final_finish_line"))

import mlpilot as ml
import seaborn as sns
from sklearn.model_selection import cross_val_score
import pandas as pd

# Load data (with offline fallback if needed)
try:
    df = sns.load_dataset("titanic")
except:
    df = pd.DataFrame({
        "survived": [0,1,1,0,0]*180,
        "pclass": [3,1,3,1,3]*180,
        "sex": ["male","female","female","female", "male"]*180,
        "age": [22,38,26,35,23]*180,
        "fare": [7.25, 71.28, 7.92, 53.10, 8.05]*180,
        "alive": ["no", "yes", "yes", "no", "no"]*180
    })

print(f"--- MLPILOT PYPI v{ml.__version__} AUDIT ---")
c = ml.clean(df, target="survived", verbose=False)
X, y = ml.features(c.df, target="survived", verbose=False)
r = ml.baseline(X, y, verbose=False)

print("alive in X:", "alive" in X.columns)
print("score:", round(r.best_score, 4))
print("CV results:", [round(s, 4) for s in cross_val_score(r.best_model, X, y, cv=5)])
print("CV mean:", round(cross_val_score(r.best_model, X, y, cv=5).mean(), 4))
