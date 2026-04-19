"""
mlpilot/ai/story.py
DataStory — Generates narrative-driven project summaries.
Uses LLMs to transform cold metrics into a readable report story.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

from mlpilot.utils.display import print_step, print_success, print_warning


class DataStory:
    """
    Narrative generator for ML projects.
    """
    def __init__(self, engine: str = "auto", model: str = "llama3", verbose: bool = False):
        self.engine = engine
        self.model = model
        self.verbose = verbose

    def tell(self, context_objects: List[Any]) -> str:
        """
        Generate a narrative story from a list of mlpilot results.
        """
        if self.verbose:
            print_step("Writing your data story...", "B ") # Use safe ASCII
        
        # 1. Gather context
        summary_text = ""
        for obj in context_objects:
            summary_text += f"\n- {repr(obj)}"
            if hasattr(obj, 'report') and hasattr(obj.report, 'changes'):
                summary_text += f" (Changes: {len(obj.report.changes)})"

        # 2. Build Narrative Prompt
        prompt = f"""
You are a lead data scientist. Write a professional, 3-paragraph executive summary based on these project metrics:
{summary_text}

STRUCTURE:
Paragraph 1: The Data Quality (mention cleaning and EDA).
Paragraph 2: The Model Performance (mention metrics and best model).
Paragraph 3: Conclusion and next steps.

STORY:
"""

        # 3. LLM Call
        res = self._call_llm(prompt)
        
        if res:
            print_success("Story generated.")
            return res
        else:
            return self._template_fallback(context_objects)

    def _call_llm(self, prompt: str) -> Optional[str]:
        # 1. Try Ollama (Local)
        try:
            import ollama
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response'].strip()
        except Exception:
            pass

        # 2. Try Gemini (Cloud Fallback 1)
        gemini_key = os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception:
                pass

        # 3. Try Groq (Cloud Fallback 2)
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                import groq
                client = groq.Groq(api_key=groq_key)
                completion = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                pass

        # 4. Try Built-in Zero-Config engine
        from mlpilot.ai.engine import call_builtin_llm
        res = call_builtin_llm(
            prompt, 
            system_prompt="You are a lead data scientist. Write a professional executive summary."
        )
        if res:
            return res

        return None

    def _template_fallback(self, context_objects: List[Any]) -> str:
        print_warning("LLM unavailable — using template fallback for story.")
        return f"Project Summary:\nThis project processed data with the following components: {[type(o).__name__ for o in context_objects]}. Quality improved and models were successfully compared."
