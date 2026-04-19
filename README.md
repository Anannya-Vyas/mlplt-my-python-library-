# mlpilot 🚀

**The complete, unbreakable machine learning toolkit — from raw data to production in seconds.**

[![PyPI Version](https://img.shields.io/pypi/v/mlplt.svg?color=blue)](https://pypi.org/project/mlplt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 The Vision: "Unbreakable & Honest Data Science"

Most Machine Learning libraries fail when things get messy—or worse, they give you "perfect" results that fall apart in production. **mlpilot** is built for the real world. It is a self-healing, zero-config environment that ensures your models are both **Unbreakable** and **Statistically Honest**.

- **Leakage Guard (v2.0.2)**: Advanced proxy detection. Automatically drops features with >0.98 correlation to the target to prevent "cheating" and ensure realistic hackathon winning scores.
- **Self-Healing 2.0**: Real-time syntax correction and "Hallucination Immunity" for AI-generated code.
- **Global Encoding Shield**: Revolutionary protection against `UnicodeEncodeError` on legacy Windows terminals.
- **Silent Production**: Absolute suppression of 3rd-party technical noise (HuggingFace, Transformers, etc.).

---

## ⚡ Quick Start: The "Omni-Pipeline"

Execute a professional, honest ML workflow in **10 lines of code**. For a detailed step-by-step guide, see our **[🚀 Hands-On Tutorial (Titanic)](TUTORIAL.md)**.

```python
import mlpilot as ml
import seaborn as sns

# 1. Load data
try:
    df = sns.load_dataset('titanic')
except:
    import pandas as pd
    df = pd.read_csv("your_data.csv")

# 2. The Unbreakable Pipeline
clean = ml.clean(df, target='survived')              # Hardens data & DROPS LEAKY PROXIES (like 'alive')
X, y  = ml.features(clean.df, target='survived')     # Zero-leakage engineering
blitz = ml.baseline(X, y)                            # Verified model tournament
tuned = ml.tune(blitz.best_name, X, y, time_budget=30) # Smart hyperparameter tuning

# 3. Defensible Verdict (Realistic accuracy, not 1.0)
print(f"Winner: {blitz.best_name} | Score: {blitz.best_score:.4f}")
```

---

## 📋 Module Encyclopedia

### 🧬 Phase 1: Data Foundations & Cleansing
*   `ml.analyze(df)`: 12-section SmartEDA report with quality scoring.
*   `ml.clean(df)`: The ultimate hardening tool. Handles nulls, outliers, dtypes, duplicates, and **Data Leakage Detection**.
*   `ml.validate(df, schema)`: Verifies your data against a strictly inferred industrial schema.

### 🧪 Phase 2: The Predictive Core
*   `ml.features(df)`: Leakage-safe fit_transform() engine. Statistics are learned from training data ONLY.
*   `ml.baseline(X, y)`: A high-speed model tournament. Compares 12+ model families to find the best baseline.
*   `ml.tune(name, X, y)`: Budget-aware hyperparameter tuning using a local optimization engine.
*   `ml.evaluate(model, X, y)`: Generates a 5-metric technical report with confusion matrices.

### 🔍 Phase 3: Specialized Insight
*   `ml.explain(model, X)`: Professional interpretability (SHAP). Tells you exactly *why* a prediction was made.
*   `ml.balance(df)`: Fixes class imbalance using intelligent oversampling (SMOTE).
*   `ml.forecast(series)`: **TimeSense** — Advanced multi-model time-series forecasting.

---

## 🧠 Advanced: The "Unbreakable" Engine

### Smart Leakage Guard (v2.0.2)
In the real world, datasets often include "cheating" columns (proxies) that are direct synonyms for the target. **mlpilot v2.0.2** introduces the **Leakage Guard**. It calculates feature-target correlation in real-time and drops variables with > 0.98 correlation. This prevents "Fake Perfect Scores" and ensures your hackathon results are trustworthy and defensible.

### AST-Healing & Hallucination Immunity
mlpilot scans every line of AI-generated code against Python standards. It **automatically purges hallucinations** and redirects file-load errors to data already in memory, ensuring zero crashes during Natural Language interactions.

### The Encoding Shield
Windows legacy terminals (CP1252) often crash when a library prints emojis. mlpilot features a **Global Encoding Shield** that automatically sanitizes all output for your specific terminal in real-time.

---

## 📦 Installation & Environments

**Standard Build (v2.0.4)**:
```bash
pip install mlplt==2.0.4
```

**GitHub Repository**:
[https://github.com/Anannya-Vyas/mlplt-my-python-library-](https://github.com/Anannya-Vyas/mlplt-my-python-library-)

---

## 🤝 Contributing & Support

MLPilot is a professional-grade machine learning toolkit. We welcome bug reports and collaborations.

- **Developer**: Anannya Vyas
- **Email**: [vyasanannya@gmail.com](mailto:vyasanannya@gmail.com)
- **GitHub**: [Anannya-Vyas/mlplt-my-python-library-](https://github.com/Anannya-Vyas/mlplt-my-python-library-)

**Current Version**: `v2.0.4` (The "Public Professional" Release)

MIT © 2026 mlpilot
