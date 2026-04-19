import urllib.request
import tarfile
import os

url = "https://files.pythonhosted.org/packages/source/m/mlplt/mlplt-2.0.2.tar.gz"
filename = "mlplt-pypi-verify.tar.gz"

print(f"--- PYPI PACKAGE SELF-AUDIT ---")
print(f"Downloading: {url}")

try:
    urllib.request.urlretrieve(url, filename)
    print("Download successful.")
    
    with tarfile.open(filename, "r:gz") as tar:
        # Check if the fix exists in strategies.py
        target_file = "mlplt-2.0.2/mlpilot/clean/strategies.py"
        try:
            f = tar.extractfile(target_file)
            if f:
                content = f.read().decode('utf-8')
                if "class LeakageGuard" in content:
                    print("\n[VERIFIED] 'class LeakageGuard' exists in the PyPI source distribution.")
                    # Show the implementation to prove it's the real one
                    start_idx = content.find("class LeakageGuard")
                    print("\n--- SNIPPET FROM PYPI SOURCE ---")
                    print(content[start_idx:start_idx+500])
                    print("--- END SNIPPET ---")
                else:
                    print("\n[FAILED] 'class LeakageGuard' NOT found in the PyPI source distribution.")
        except KeyError:
            print(f"\n[FAILED] Could not find {target_file} in the archive.")

    # Cleanup
    os.remove(filename)

except Exception as e:
    print(f"\n[ERROR] Verification failed: {e}")
