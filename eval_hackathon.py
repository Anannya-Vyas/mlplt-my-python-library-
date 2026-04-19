import os
import sys
import warnings

# Suppress warnings for a clean report
warnings.filterwarnings("ignore")

def run_evaluation():
    print("--- MLPILOT HACKATHON READINESS AUDIT ---")
    print("-" * 40)
    
    scorecard = {
        "Environment Check": False,
        "Data Load": False,
        "Auto Clean": False,
        "Feature Forge": False,
        "Baseline Blitz": False
    }
    
    try:
        # 1. Environment Check
        import mlpilot as ml
        import seaborn as sns
        import pandas as pd
        scorecard["Environment Check"] = True
        print("[OK] Step 1: Environment (mlpilot, seaborn, pandas) loaded.")
        
        # 2. Data Load
        try:
            df = sns.load_dataset('titanic')
        except Exception:
            # Fallback for offline environments
            print("[INFO] Seaborn load failed. Using manual Titanic creation...")
            df = pd.DataFrame({
                'survived': [0, 1, 1, 0, 0] * 20,
                'pclass': [3, 1, 3, 1, 3] * 20,
                'sex': ['male', 'female', 'female', 'female', 'male'] * 20,
                'age': [22, 38, 26, 35, 23] * 20,
                'fare': [7.25, 71.28, 7.92, 53.10, 8.05] * 20
            })
            
        if not df.empty:
            scorecard["Data Load"] = True
            print(f"[OK] Step 2: Titanic data loaded ({len(df)} rows).")
        
        # 3. Auto Clean
        cleaned = ml.clean(df, target='survived', verbose=False)
        if not cleaned.df.empty:
            scorecard["Auto Clean"] = True
            print("[OK] Step 3: Auto-cleaning successful.")
        
        # 4. Feature Forge
        X, y = ml.features(cleaned.df, target='survived', verbose=False)
        if X.shape[1] > 0:
            scorecard["Feature Forge"] = True
            print(f"[OK] Step 4: Feature engineering success ({X.shape[1]} columns).")
            
        # 5. Baseline Blitz (The Big One)
        result = ml.baseline(X, y, verbose=False)
        if result and result.best_score > 0:
            scorecard["Baseline Blitz"] = True
            print(f"[OK] Step 5: Training success. Best model: {result.best_name} ({result.best_score:.4f})")

        print("\n" + "="*40)
        print(" FINAL READINESS SCORECARD ")
        print("="*40)
        all_passed = True
        for task, status in scorecard.items():
            icon = "[PASS]" if status else "[FAIL]"
            if not status: all_passed = False
            print(f"{icon} {task:<20}")
        
        if all_passed:
            print("\nVERDICT: 100% READY FOR HACKATHON")
        else:
            print("\nVERDICT: SYSTEM CRITICAL - NOT READY")

    except Exception as e:
        print(f"\n[CRASH] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()
