# mlpilot — The Complete ML Toolkit
### A Hands-On Tutorial by Anannya Vyas

Welcome! This notebook walks you through **mlpilot**, a Python library that takes you from **raw messy data → trained ML model → explanations** in just a few lines of code.

**What you will learn:**
- How to clean messy real-world data automatically
- What "data leakage" is and how mlpilot guards against it  
- How to train and compare 13+ ML models in seconds
- How to understand *why* your model makes predictions
- How to forecast time series and generate deployment files

**No prior ML library experience needed — every step is explained.**

---
- **PyPI:** https://pypi.org/project/mlplt/  
- **GitHub:** https://github.com/Anannya-Vyas/mlplt-my-python-library-  
- **Author:** Anannya Vyas · vyasanannya@gmail.com

## Step 1 — Install mlpilot

The install name is `mlplt` but we import it as `mlpilot` (same pattern as `pip install scikit-learn` → `import sklearn`).

```python
!pip install mlplt==2.0.4 --quiet
print("Installation complete!")
```

## Step 2 — Import the Library

We import it as `ml` for short — so all functions look like `ml.clean()`, `ml.baseline()`, etc.

```python
import mlpilot as ml
import sns
import pandas as pd

print("mlpilot version:", ml.__version__)
print("Ready!")
```

## Step 3 — Load a Dataset

We'll use the famous **Titanic dataset** — predicting who survived based on passenger information (age, class, gender, ticket price, etc.).

This dataset is a good example of real-world messy data: it has **missing values**, **mixed data types**, and even a sneaky **"cheating" column** (we'll expose that soon!).

```python
df = sns.load_dataset('titanic')
print(f"Shape: {df.shape}  ({df.shape[0]} passengers, {df.shape[1]} columns)")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
df.head()
```

## Step 4 — Automatic Data Analysis (EDA)

Before touching the data, let's understand it. `ml.analyze()` generates a full quality report automatically — no manual `df.describe()` loops needed.

```python
report = ml.analyze(df)
# This prints distributions, missing value counts, correlations, and a quality score
```

## Step 5 — Clean the Data

Real-world data is messy. `ml.clean()` handles:
- **Missing values** — fills or drops intelligently  
- **Duplicates** — removes them  
- **Outliers** — detects and clips extreme values  
- **Type mismatches** — fixes mixed dtypes  
- **Data leakage** — this is the special one, explained below

### What is data leakage?
The Titanic dataset has an `alive` column — which is literally just another way of saying `survived`. If we trained a model using it, we'd get **100% accuracy**, but the model learned nothing. It just copied the answer.

`mlpilot` automatically detects and removes such columns.

> **Watch the output below for:** `"Dropped X leaky proxy columns"` — that's the Leakage Guard in action.

```python
clean = ml.clean(df, target='survived')
print("\nCleaned data shape:", clean.df.shape)
print("'alive' still in columns?", 'alive' in clean.df.columns)
clean.df.head()
```

### Why does this matter?

| | Without Leakage Guard | With Leakage Guard |
|---|---|---|
| `alive` column | Included | **Automatically dropped** |
| Model score | 1.0000 (fake perfect) | **0.8557 (realistic)** |
| Defensible? | No | **Yes** |

A score of **1.0000** on any real dataset is almost always a bug, not a breakthrough. If you ever see this, check for leaky columns.

## Step 6 — Feature Engineering

ML models only understand numbers, not text. `ml.features()` automatically:
- Converts text columns like `"male"/"female"` → `0/1`
- Scales numeric features to similar ranges
- Returns `X` (features) and `y` (target) ready for training

```python
X, y = ml.features(clean.df, target='survived')
print(f"Features (X) shape: {X.shape}")
print(f"Target  (y) shape:  {y.shape}")
print(f"Feature columns: {list(X.columns)}")
X.head()
```

## Step 7 — Train & Compare 13+ Models

Normally you'd try each model one by one — Logistic Regression, Random Forest, Gradient Boosting... This takes hours.

`ml.baseline()` runs all of them in parallel using **cross-validation** and gives you a ranked leaderboard in under 60 seconds.

```python
import time
start = time.time()

blitz = ml.baseline(X, y)

elapsed = round(time.time() - start, 1)
print(f"\nTime taken: {elapsed}s")
print(f"Best model: {blitz.best_name}")
print(f"Best score: {round(blitz.best_score, 4)}")
```

### Understanding the Score

- **~0.85** = excellent for Titanic (industry standard is 0.78–0.85)
- **~0.70** = decent, room to improve
- **1.0000** = something is wrong (data leakage!)

The score is a **cross-validation score** — this means it was tested on data the model never saw during training, making it a trustworthy estimate of real-world performance.

## Step 8 — Evaluate the Model

Get a full technical report: accuracy, precision, recall, F1-score, and a confusion matrix.

```python
report = ml.evaluate(blitz.best_model, X, y)
print(report)
```

## Step 9 — Hyperparameter Tuning (Optional)

Every ML model has "knobs" (hyperparameters) that can be adjusted. `ml.tune()` automatically searches for the best settings within a time budget you set.

```python
# time_budget=30 means: try different settings for 30 seconds, return the best one
try:
    tuned = ml.tune(blitz.best_name, X, y, time_budget=30)
    print(f"Tuned score: {round(tuned.best_score, 4)}")
    improvement = round((tuned.best_score - blitz.best_score) * 100, 2)
    print(f"Improvement over baseline: {improvement}%")
except Exception as e:
    print(f"Tuning skipped: {e}")
    print("Install with: pip install mlplt[optuna]")
```

## Step 10 — Explain the Model

A model that just says "yes/no" isn't useful in the real world — people need to know **why**. `ml.explain()` uses SHAP (a standard industry technique) to show which features pushed a prediction up or down.

```python
try:
    exp = ml.explain(blitz.best_model, X)
    exp.feature_importance()  # Shows which columns matter most
except Exception as e:
    print(f"Explain skipped: {e}")
    print("Install with: pip install mlplt[shap]")
```

## Step 11 — Time Series Forecasting (Bonus)

mlpilot also handles time series data — predicting future values based on historical patterns. Useful for sales forecasting, stock trends, weather, etc.

```python
import numpy as np

# Create a sample time series (replace with your own data)
dates = pd.date_range('2023-01-01', periods=100)
series = pd.DataFrame({
    'ds': dates,
    'y': np.random.randn(100).cumsum() + 50  # simulated trend
})
print("Sample time series:")
print(series.tail())

try:
    forecast_result = ml.forecast(series)
    print("\nForecast result:")
    print(forecast_result)
except Exception as e:
    print(f"Forecast skipped: {e}")
    print("Install with: pip install mlplt[prophet]")
```

## Step 12 — Generate Deployment Files

Turn your model into a production API with one command. `ml.deploy()` generates a FastAPI application and a Dockerfile that any engineering team can run.

```python
import os, shutil

try:
    deployment = ml.deploy(blitz.best_model)
    print("Deployment files generated!")
    if os.path.exists("ml_api"):
        print("Files created in ml_api/:")
        for f in os.listdir("ml_api"):
            print(f"  {f}")
        shutil.rmtree("ml_api")  # Clean up for this demo
except Exception as e:
    print(f"Deploy info: {e}")
```

## Step 13 — Use Your Own Data

Replace the Titanic dataset with any CSV you have. The whole pipeline stays the same — just change the filename and target column name.

```python
# TEMPLATE — replace with your own data
# df = pd.read_csv("your_data.csv")
# clean = ml.clean(df, target="your_target_column")
# X, y  = ml.features(clean.df, target="your_target_column")
# blitz = ml.baseline(X, y)
# print(f"Best model: {blitz.best_name} | Score: {blitz.best_score:.4f}")
```

print("Ready to use with your own data!")
print("Just replace 'your_data.csv' and 'your_target_column' above.")

## Summary — What You Just Did

In about **15 lines of code** and under 5 minutes, you ran a full industry-grade ML workflow:

| Step | Function | What it did |
|------|----------|-------------|
| 1 | `ml.analyze()` | Understood the raw data |
| 2 | `ml.clean()` | Fixed nulls, outliers, leakage |
| 3 | `ml.features()` | Encoded and scaled features |
| 4 | `ml.baseline()` | Compared 13+ models |
| 5 | `ml.evaluate()` | Got a full performance report |
| 6 | `ml.tune()` | Optimized the best model |
| 7 | `ml.explain()` | Understood the predictions |
| 8 | `ml.forecast()` | Predicted a time series |
| 9 | `ml.deploy()` | Generated a production API |

---

### Next Steps
- Try it on your own dataset!
- Explore: `ml.balance()`, `ml.sentiment()`, `ml.topics()`, `ml.validate()`
- Read the full docs: https://pypi.org/project/mlplt/

---
**Made by Anannya Vyas** · vyasanannya@gmail.com · [GitHub](https://github.com/Anannya-Vyas/mlplt-my-python-library-)
