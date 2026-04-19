"""
mlpilot/utils/setup_guide.py
Guided setup experience for mlpilot.
"""

from __future__ import annotations

import os
from mlpilot.utils.env import get_platform_info, is_colab
from mlpilot.utils.display import print_banner, print_step, print_success, print_warning


def setup():
    """
    Run a guided setup to ensure mlpilot is ready for your environment.
    Checks dependencies, environment, and AI connectivity.
    """
    info = get_platform_info()
    
    print_banner("mlpilot Setup Guide")
    print_step(f"Environment: {info['os']} ({'Cloud/Colab' if info['is_colab'] else 'Local'})")
    print_step(f"Python: {info['python_version']}\n")

    # 1. Dependency Checks
    _check_ai_deps()
    
    # 2. Environment Specifics
    if info['is_colab']:
        _colab_specific_setup()
    else:
        _local_specific_setup()

    # 3. AI Connectivity
    _check_ai_connectivity()

    print_success("\nSetup check complete! You're ready to fly. 🚀")


def _check_ai_deps():
    print_step("Checking optional dependencies...", "📦")
    missing = []
    
    try: import ollama
    except ImportError: missing.append("ollama")
    
    try: import groq
    except ImportError: missing.append("groq")

    try: import google.generativeai
    except ImportError: missing.append("google-generativeai")

    if missing:
        print_warning(f"Missing optional AI components: {', '.join(missing)}")
        print_step("To fix, run:", "💡")
        print(f"  pip install mlpilot[ai] {' '.join(missing)}")
    else:
        print_success("All AI dependencies found.")


def _colab_specific_setup():
    print_step("Google Colab detected.", "☁️")
    print_step("To use local-LLM features in Colab, you need to start Ollama.", "💡")
    print_step("Run: ml.colab_setup()", "🛠️")


def _local_specific_setup():
    print_step("Local environment detected.", "💻")
    # Check for Ollama service
    try:
        import ollama
        ollama.list()
        print_success("Local Ollama service is running.")
    except Exception:
        print_warning("Local Ollama service not detected.")
        print_step("Download from https://ollama.com to use local-first AI features.", "🌍")


def _check_ai_connectivity():
    print_step("Checking LLM API Keys...", "🔑")
    keys = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }
    
    found_any = False
    for name, key in keys.items():
        if key:
            print_success(f"{name} is configured.")
            found_any = True
        else:
            print(f"  [ ] {name} not found.")

    if not found_any:
        print_step("No cloud API keys found. mlpilot will default to local Ollama.", "ℹ️")
        print_step("To add a cloud fallback, get a free key:", "🎁")
        print("  - Gemini: https://aistudio.google.com/")
        print("  - Groq:   https://console.groq.com/")
        print("\n  Set them as environment variables or pass them to ml.analyst(df, key=...)")
