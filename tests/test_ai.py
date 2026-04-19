"""
tests/test_ai.py
Tests for AI Differentiators (AIAnalyst, MLAudit, DataStory).
Uses mocking to simulate LLM providers.
"""

import sys
from unittest.mock import MagicMock, patch

# Mocking missing optional dependencies for testing
sys.modules["ollama"] = MagicMock()
sys.modules["groq"] = MagicMock()
sys.modules["fairlearn"] = MagicMock()
sys.modules["fairlearn.metrics"] = MagicMock()

import numpy as np
import pandas as pd
import pytest
import mlpilot as ml


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 22],
        "income": [50000, 60000, 70000, 80000, 45000],
        "gender": ["M", "F", "F", "M", "M"],
        "target": [0, 1, 1, 0, 0]
    })


class TestAIAnalyst:
    @patch("ollama.generate")
    @patch("ollama.list")
    @patch("builtins.input", return_value="y")
    def test_analyst_ask_ollama(self, mock_input, mock_list, mock_generate, sample_df):
        # Mocking Ollama
        mock_list.return_value = {"models": [{"name": "llama3"}]}
        mock_generate.return_value = {"response": "result = df['age'].mean()"}
        
        analyst = ml.analyst(sample_df, engine="ollama", verbose=False)
        res = analyst.ask("What is the average age?")
        
        assert res == 30.4
        mock_generate.assert_called_once()
        mock_input.assert_called_once()

    @patch("groq.Groq")
    @patch("builtins.input", return_value="y")
    def test_analyst_ask_groq(self, mock_input, mock_groq_class, sample_df):
        # Mocking Groq
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "result = df['income'].max()"
        
        analyst = ml.analyst(sample_df, engine="groq", groq_api_key="sk-test", verbose=False)
        res = analyst.ask("Max income?", auto_run=False)
        
        assert res == 80000
        mock_client.chat.completions.create.assert_called_once()

    def test_analyst_no_engine_warning(self, sample_df):
        # Should not crash if no engine found
        with patch("mlpilot.ai.analyst.AIAnalyst._is_ollama_alive", return_value=False):
            analyst = ml.analyst(sample_df, engine="auto", verbose=False)
            assert analyst.engine in ["groq", "bash"]


class TestMLAudit:
    def test_audit_technical(self, sample_df):
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 1, 0, 0])
        
        res = ml.audit(model, sample_df.drop("target", axis=1), sample_df["target"], verbose=False)
        assert res.technical["stability"] >= 0
        assert res.model_info["type"] == "MagicMock"

    def test_audit_fairness(self, sample_df):
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 1, 0, 0])
        
        # Test with sensitive features
        res = ml.audit(model, sample_df.drop("target", axis=1), sample_df["target"], 
                       sensitive_features=sample_df["gender"], verbose=False)
        
        # If fairlearn is mocked/missing, it should still run but might have empty fairness dict
        assert isinstance(res.fairness, dict)


class TestDataStory:
    @patch("ollama.generate")
    def test_story_tell(self, mock_generate):
        mock_generate.return_value = {"response": "This is a great story about the data."}
        
        # Mock result objects
        results = [MagicMock(), MagicMock()]
        story_text = ml.story(results, verbose=False)
        
        assert "story" in story_text.lower()
        mock_generate.assert_called_once()
