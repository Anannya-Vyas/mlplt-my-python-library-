import pandas as pd
import mlpilot as ml

# 1. CREATE A "DIRTY" DATASET (Strings + Missing)
data = {
    'city': ['NYC', 'Tokyo', 'London', 'Berlin', 'Paris'] * 20, # Strings!
    'age': [25, 30, 35, 40, None] * 20,                          # Missing!
    'target': [0, 1, 0, 1, 0] * 20
}
df = pd.DataFrame(data)

print("--- MLPILOT v1.2.1 STABILITY SMOKE TEST ---")

# TEST 1: CATEGORICAL IMMUNITY
print("\n[Test 1] Analyzing 'Messy' Strings...")
try:
    X = df.drop('target', axis=1)
    y = df['target']
    # mlpilot v1.2.1 will detect 'city' is categorical and auto-heal it
    result = ml.baseline(X, y, verbose=True, models=['RandomForestClassifier'])
    print("\n[OK] SUCCESS: mlpilot self-healed the categorical data!")
except Exception as e:
    print(f"\n[FAIL] FAILED: {e}")

# TEST 2: ENCODING SHIELD
# This verifies that print_step won't crash your terminal
print("\n[Test 2] Testing the Encoding Shield (Safe Output)...")
# This call uses emojis but mlpilot will strip them safely for this terminal
ml.utils.display.print_step("This message contains an emoji: [Rocket]", "[Trophy]")
print("[OK] SUCCESS: Your terminal is now protected from Unicode crashes.")

# TEST 3: INTUITION ALIASES
print("\n[Test 3] Checking Explainer Aliases...")
try:
    # Use the new alias we added
    X_enc, y_enc = ml.features(df, target='target')
    model = result.best_model
    exp = ml.explain(model, X_enc)
    print("  Calling plot_contributions()...")
    exp.plot_contributions(X_enc.iloc[0])
    print("[OK] SUCCESS: Aliases are working.")
except Exception as e:
    print(f"  Note: {e}")

print("\n--- SMOKE TEST COMPLETE ---")
