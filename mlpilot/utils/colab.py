"""
mlpilot/utils/colab.py
Google Colab specific utilities (Ollama setup, etc.)
"""

from __future__ import annotations

import os
import subprocess
import time
import sys
from mlpilot.utils.display import print_step, print_success, print_error, print_warning


def is_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available in the current environment."""
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_ollama(model_to_pull: str = "llama3.2", verbose: bool = False):
    """
    Silent background setup for Ollama in Google Colab.
    """
    if verbose:
        print_step(f"Setting up local AI ({model_to_pull})...", "🤖")
    
    try:
        # 1. Install dependencies silently
        subprocess.run(["apt-get", "update"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "pciutils"], capture_output=True)

        # 2. Run install script
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, capture_output=True)
        
        # 3. Start server
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        
        # 4. Pull the model
        subprocess.run(["ollama", "pull", model_to_pull], capture_output=True)
        
        if verbose:
            print_success("AI engine ready.")
    except Exception:
        pass # Fail silently so the fallback logic can take over


def install_dependencies():
    """Install all optional dependencies needed for AI features."""
    print_step("Installing mlpilot[ai] dependencies...", "📦")
    # Use -q for quieter installation in notebooks
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlpilot[ai]"], check=True)
    print_success("Dependencies installed.")
