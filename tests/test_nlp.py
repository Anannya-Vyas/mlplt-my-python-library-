"""
tests/test_nlp.py
Tests for TextML (NLP toolkit).
"""

import pandas as pd
import pytest
import mlpilot as ml


@pytest.fixture
def text_df():
    return pd.DataFrame({
        "text": [
            "This was an absolutely great movie!",
            "Terrible experience, would not recommend.",
            "Average at best, nothing special.",
            "Loved the pacing and the acting was top notch.",
            "Worst service ever, stay away.",
            "The food was okay, but the price was too high.",
        ],
        "label": [1, 0, 0, 1, 0, 0]
    })


class TestTextML:
    def test_sentiment_ranges(self, text_df):
        scores = ml.sentiment(text_df["text"], verbose=False)
        assert len(scores) == len(text_df)
        assert all(-1 <= s <= 1 for s in scores)
        # Check basic polarity
        assert scores.iloc[0] > 0   # "great"
        assert scores.iloc[1] < 0   # "terrible"

    def test_text_classify_runs(self, text_df):
        result = ml.text_classify(text_df, text_col="text", target="label", verbose=False)
        assert result.model is not None
        # Predict on same data for smoke test
        preds = result.model.predict(text_df["text"])
        assert len(preds) == len(text_df)

    def test_topics_discovery(self, text_df):
        result = ml.topics(text_df["text"], n_topics=2, verbose=False)
        assert len(result.topics) == 2
        assert len(result.topics[0]) > 0

    def test_embed_shape(self, text_df):
        df_vec = ml.embed(text_df["text"], verbose=False)
        assert isinstance(df_vec, pd.DataFrame)
        assert len(df_vec) == len(text_df)
        assert df_vec.shape[1] > 0

    def test_sentiment_fallback_robustness(self):
        # Test with empty strings and non-strings
        s = pd.Series(["", None, 123, "happy"])
        scores = ml.sentiment(s, verbose=False)
        assert len(scores) == 4
        assert scores.iloc[3] > 0
