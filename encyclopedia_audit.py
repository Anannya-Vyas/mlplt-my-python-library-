import pandas as pd
import numpy as np
import mlpilot as ml
import os
import shutil

def run_audit():
    ml.colab_setup()
    ml.utils.display.print_banner("mlpilot v1.2.2: ENCYCLOPEDIA AUDIT")

    # PREPARE DATASETS
    df = pd.DataFrame({
        'target': [0, 1, 0, 1, 0] * 20,
        'age': [20, 30, 40, 50, None] * 20,
        'city': ['NYC', 'Tokyo', 'London', 'Berlin', 'Paris'] * 20,
        'date': pd.date_range('2023-01-01', periods=100),
        'text': ["I love this library!", "It is okay.", "Not great.", "Amazing!", "A bit complex."] * 20
    })

    # =======================================================================
    # PHASE 1: DATA FOUNDATIONS
    # =======================================================================
    print("\n--- PHASE 1: DATA FOUNDATIONS ---")
    
    # 1. analyze()
    ml.analyze(df, verbose=False)
    print("[OK] ml.analyze() executed.")

    # 2. clean()
    cleaned = ml.clean(df)
    print("[OK] ml.clean() executed.")

    # 3. validate() & infer_schema()
    schema = ml.infer_schema(df)
    ml.validate(df, schema)
    print("[OK] ml.validate() & ml.infer_schema() executed.")

    # 4. features()
    X, y = ml.features(cleaned.df, target='target')
    print("[OK] ml.features() executed.")

    # 5. balance()
    ml.balance(cleaned.df, target='target')
    print("[OK] ml.balance() executed.")

    # =======================================================================
    # PHASE 2: TRAINING ENGINE
    # =======================================================================
    print("\n--- PHASE 2: TRAINING ENGINE ---")

    # 6. baseline()
    board = ml.baseline(X, y, verbose=False, models=['LogisticRegression', 'DecisionTreeClassifier'])
    print("[OK] ml.baseline() executed.")

    # 7. evaluate()
    ml.evaluate(board.best_model, X, y)
    print("[OK] ml.evaluate() executed.")

    # 8. tune()
    ml.tune('LogisticRegression', X, y, time_budget=5)
    print("[OK] ml.tune() executed.")

    # 9. explain()
    try:
        ml.explain(board.best_model, X)
        print("[OK] ml.explain() executed.")
    except Exception as e:
        print(f"[NOTE] ml.explain() skipped/handled: {e}")

    # =======================================================================
    # PHASE 3: SPECIALIZED INTELLIGENCE
    # =======================================================================
    print("\n--- PHASE 3: SPECIALIZED ---")

    # 10. forecast()
    ts_df = df[['date', 'age']].rename(columns={'date': 'ds', 'age': 'y'}).dropna()
    try:
        ml.forecast(ts_df)
        print("[OK] ml.forecast() executed.")
    except Exception as e:
        print(f"[NOTE] ml.forecast() skipped/handled: {e}")

    # 11. NLP Suite
    try:
        # sentiment
        ml.sentiment(df['text'].head(5))
        # topics
        ml.topics(df['text'].head(5))
        print("[OK] ml.sentiment() & ml.topics() executed.")
    except Exception as e:
        print(f"[NOTE] ml.nlp suite skipped/handled: {e}")

    # =======================================================================
    # PHASE 4: PRODUCTION & AI
    # =======================================================================
    print("\n--- PHASE 4: PRODUCTION & AI ---")

    # 12. audit()
    ml.audit(board.best_model, X, y)
    print("[OK] ml.audit() executed.")

    # 13. story()
    ml.story().tell([{'model': 'X', 'score': 0.9}])
    print("[OK] ml.story() executed.")

    # 14. deploy()
    try:
        ml.deploy(board.best_model)
        # Cleanup mock deployment files
        if os.path.exists("ml_api"): shutil.rmtree("ml_api")
        print("[OK] ml.deploy() executed.")
    except Exception as e:
        print(f"[NOTE] ml.deploy() skipped/handled: {e}")

    # 15. session()
    with ml.session("audit_session") as sess:
        ml.analyze(df.head(10), verbose=False)
        sess.export_script("audit_session.py")
    
    if os.path.exists("audit_session.py"): os.remove("audit_session.py")
    print("[OK] ml.session() executed.")

    # 16. analyst()
    try:
        a = ml.analyst(df)
        # Package Hijack check
        a.ask("What is the average age?", auto_run=True)
        print("[OK] ml.analyst() executed.")
    except Exception as e:
        print(f"[NOTE] ml.analyst() skipped/handled: {e}")

    print("\n--- ENCYCLOPEDIA AUDIT COMPLETE ---")

if __name__ == "__main__":
    run_audit()
