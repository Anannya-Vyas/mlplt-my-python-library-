# mlpilot 🚀

**The complete, unbreakable machine learning toolkit — from raw data to production in seconds.**

[![PyPI Version](https://img.shields.io/pypi/v/mlplt.svg?color=blue)](https://pypi.org/project/mlplt/)


---

## 🌟 The Vision: "Unbreakable Data Science"

Most Machine Learning libraries fail when things get messy. **mlpilot** is the first library built to be **Unbreakable**. It is designed for the real world—where data has typos, models hallucinate, and servers have legacy encodings. 

**One Import. Every Tool.**  
Whether you are a student in Google Colab, a Data Scientist on a local laptop, or a Production Engineer deploying to Docker, mlpilot provides a self-healing, zero-config environment that just works.

- **Self-Healing 2.0**: Real-time syntax correction, "Hallucination Immunity," and now **Categorical Healing** for raw string features.
- **Global Encoding Shield**: Revolutionary protection against `UnicodeEncodeError` on legacy Windows terminals (CP1252/ASCII).
- **Silent Production**: Absolute suppression of 3rd-party technical noise and clutter.

---

## ⚡ Quick Start: The "Omni-Pipeline"

Execute a professional, industrial ML workflow in **10 lines of code**:

```python
import mlpilot as ml
import seaborn as sns

# 1. Load data
df = sns.load_dataset('titanic')

# 2. The Unbreakable Pipeline
clean = ml.clean(df, target='survived')              # Hardens data against noise
X, y  = ml.features(clean.df, target='survived')     # Zero-leakage engineering
blitz = ml.baseline(X, y)                            # 10s "Winner-Takes-All" model search
tuned = ml.tune(blitz.best_name, X, y, time_budget=30) # Smart hyperparameter tuning
audit = ml.audit(tuned.best_model, X, y)             # Fairness & Bias verification

# 3. AI Insights
ml.analyst(df).ask("What is the survival rate for females?", auto_run=True)
```

---

## 📋 The Clear Vision: Why mlpilot?

| Challenge | The Hard Way (Manual) | **The mlpilot Way** | Real Impact |
| :--- | :--- | :--- | :--- |
| **Data Cleaning** | 50+ lines of loops, imputers, and drops. | `ml.clean(df)` | **98% less code.** No logic errors. |
| **EDA Insights** | 100+ lines of `matplotlib` and `seaborn`. | `ml.analyze(df)` | **Professional reports** in 5 seconds. |
| **Model Search** | Manually trying 10+ algorithms in a loop. | `ml.baseline(X, y)` | Finds the **winner** in 10 seconds. |
| **Deep Tuning** | Complex `GridSearchCV` or `Optuna` setup. | `ml.tune(name, X, y)` | **Best hyperparams** with zero math. |
| **Deployment** | 200+ lines (FastAPI, Docker, Pickling). | `ml.deploy(model)` | **Production-ready API** in 1 line. |

---

## 🛠️ The Module Encyclopedia

### 🧬 Phase 1: Data Foundations & Cleansing
Master your raw data before it hits the model.
*   `ml.analyze(df)`: Generates a 12-section SmartEDA report with automatic leakage detection and quality scoring.
*   `ml.clean(df)`: The ultimate hardening tool. Automatically handles missing values, outliers, duplicates, and inconsistent categorical dtypes.
*   `ml.validate(df, schema)`: Industrial-grade data validation. Refreshes and verifies your data against a strictly inferred schema.

### 🧪 Phase 2: The Predictive Core
Turn raw tables into high-performance predictions.
*   `ml.features(df)`: Zero-leakage automated feature engineering. Handles target encoding, scaling, and rolling aggregations.
*   `ml.baseline(X, y)`: A high-speed model search tournament. Compares 12+ model families to find the best baseline.
*   `ml.tune(name, X, y)`: Smart, budget-aware hyperparameter tuning using a local optimization engine.
*   `ml.evaluate(model, X, y)`: Generates a 5-metric technical report with confusion matrices and performance charts.

### 🔍 Phase 3: specialize & specialized Insight
Beyond simple accuracy—understand and improve your models.
*   `ml.explain(model, X)`: Professional local/global interpretability (SHAP). Tells you exactly *why* a prediction was made.
*   `ml.balance(df)`: Fixes class imbalance using intelligent oversampling (SMOTE) and undersampling techniques.
*   `ml.forecast(series)`: **TimeSense** — Advanced multi-model time-series forecasting for finance and trends.

### 🎭 Phase 4: NLP & Text Intelligence
State-of-the-art Natural Language processing with zero training required.
*   `ml.sentiment(series)`: Zero-shot emotion and sentiment extraction from raw text.
*   `ml.topics(series)`: Automatically extracts core topics and themes from thousands of comments.
*   `ml.embed(series)`: Generates state-of-the-art vector embeddings for semantic search and clustering.

### 🤖 Phase 5: AI, Audit & Production
The future of Data Science—automated, ethical, and deployable.
*   **AI Analyst**: Natural Language Interface. Ask complex questions about your data in plain English.
*   **MLAudit**: Comprehensive bias and fairness auditing. Generates automated **Model Cards** to ensure your AI is ethical.
*   **Data Story**: Merges EDA and Performance results into a professional narrative report for executives.
*   **LaunchPad**: Generates a production-ready FastAPI application and Dockerfile for any model in one command.
*   **Session**: The ultimate reproducibility tool. Records your entire workflow as a clean `.py` or `.ipynb` script.

---

## 🎯 Real-World Scenarios: Where mlpilot Shines

1.  **"The Messy CSV"**: You have a dataset with typos, missing values, and weird outliers, but your manager needs a baseline model *today*.
    *   **Solution**: `ml.clean(df)` -> `ml.baseline(X, y)` -> Done in 5 minutes.
2.  **"The Ethical Challenge"**: You need to prove to a compliance team that your model isn't biased against a specific city or gender.
    *   **Solution**: `ml.audit(model, X, y, sensitive_features=['city'])` -> Instant Model Card.
3.  **"The NLP Shortcut"**: You have 10,000 customer reviews and no time to train a classifier.
    *   **Solution**: `ml.sentiment(reviews)` -> Instant insights.
4.  **"The Production Emergency"**: You have a winning model and need to hand over an API to the engineering team.
    *   **Solution**: `ml.deploy(model)` -> Production code generated instantly.

---

## 🧠 Advanced: The "Unbreakable" Engine

What makes mlpilot different? We don't just "wrap" libraries; we harden them at the core.

### AST-Healing (Self-Healing AI)
In v1.1.5, we introduced the **Abstract Syntax Tree (AST) Scanner**. When the AI Analyst generates code, it often includes conversational noise or syntax errors. mlpilot scans every line against Python standards and **automatically purges hallucinations** before they cause a crash.

### Hallucination Immunity (v1.1.7)
AI models often hallucinate "setup" steps, like trying to load a file named `dataset.csv`. mlpilot intercepts these errors and **surgically redirects** the execution to the high-performance dataframe already in your memory.

### Categorical Immunity (v1.2.0)
Most models crash when they see "NYC" instead of `[1.0, 0.0]`. mlpilot v1.2.0 introduces **Categorical Guard**. Functions like `baseline()` and `tune()` now automatically detect, clean, and encode string data on the fly. It just works.

### The Encoding Shield (v1.2.1)
Windows legacy terminals (CP1252) often crash when a library prints an emoji (like 🚀). mlpilot v1.2.1 features a **Global Encoding Shield** that automatically sanitizes all output for your specific terminal in real-time, ensuring zero crashes.

### Atomic Silence
We have injected a **Global Silence Guard** into the core initialization. No more progress-bar spam from HuggingFace, no more technical warnings from Transformers. You get only the results you asked for.

---

## 📦 Installation & Environments

**Standard Build**:
```bash
pip install mlplt
```

**Full AI Build (Recommended)**:
```bash
pip install mlplt[full]
```

**Google Colab Setup**:
```python
import mlpilot as ml
ml.colab_setup() # Automatically configures local AI backends for 100% reliability
```

---

## 🤝 Contributing & Support

MLPilot is maintained as a high-performance, private-source toolkit to ensure stability and security. We welcome feedback, bug reports, and professional collaborations.

- **Direct Support & Contributions**: If you have an update or wish to contribute to the project, please contact us directly via email.
- **Email**: [vyasanannya@gmail.com](mailto:vyasanannya@gmail.com)

**Current Version**: `v1.2.7` (The "Unbreakable" Release)

MIT © 2026 mlpilot
