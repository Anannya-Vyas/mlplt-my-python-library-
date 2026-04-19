"""
mlpilot/ai/analyst.py
AIAnalyst — Natural language data analysis using local-first LLMs.
Uses Ollama (local) via its API, with Groq as a cloud fallback.
Implements 'Review before Run' for execution safety.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mlpilot.utils.display import print_line, print_step, print_success, print_warning
from mlpilot.utils.imports import get_ds_workspace

from mlpilot.ai.engine import call_builtin_llm


class AIAnalyst:
    """
    AI-powered data analyst companion.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model: str = "auto",
        engine: str = "auto",
        groq_api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.df = df
        self.verbose = verbose
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        
        # 1. Engine Detection
        if engine == "auto":
            if self._is_ollama_alive():
                self.engine = "ollama"
            else:
                # ZERO-CONFIG: No need for warnings, just use the built-in brain
                self.engine = "builtin"
        else:
            self.engine = engine

        # 2. Model Detection
        if self.engine == "builtin":
            self.model = "Qwen/Qwen2.5-0.5B-Instruct"
        elif model == "auto" and self.engine == "ollama":
            self.model = self._detect_best_ollama_model()
        elif model == "auto" and self.engine == "gemini":
            self.model = "gemini-1.5-flash"
        elif model == "auto" and self.engine == "groq":
            self.model = "llama3-70b-8192"
        else:
            self.model = model

    def ask(self, query: str, auto_run: bool = False) -> Any:
        """
        Ask a natural language question about the data.
        Returns the result of the generated code execution.
        """
        if self.verbose:
            print_step(f"AI Analyst ({self.engine}): Thinking about '{query}'...", "🧠")

        # 1. Build Prompt
        prompt = self._build_prompt(query)

        # 2. Get Code
        code = self._get_llm_code(prompt)
        if not code:
            print_warning("AI could not generate valid code.")
            return None

        # 3. Review before Run
        if not auto_run:
            if not self._user_approval(code):
                print_step("Execution cancelled by user.", "🚫")
                return None

        # 4. Execute
        if self.verbose:
            print_step("Executing analysis...", "⚡")
        
        return self._execute_code(code)

    def _build_prompt(self, query: str) -> str:
        try:
            head_str = self.df.head(3).to_markdown()
        except (ImportError, ModuleNotFoundError):
            head_str = str(self.df.head(3))

        summary = {
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "head": head_str,
            "shape": self.df.shape
        }
        
        prompt = f"""
You are a world-class Python Data Analyst. Your goal is to write code using pandas to answer questions about a dataframe named 'df'.

DATASET SUMMARY:
- Shape: {summary['shape']}
- Columns: {summary['columns']}
- Dtypes: {summary['dtypes']}
- Sample Data:
{summary['head']}

USER QUESTION:
"{query}"

RULES:
1. Only return the Python code itself. 
2. Do not include markdown '```python' blocks.
3. Use the dataframe variable 'df'.
4. Ensure the last line of the code results in the final answer (or stores it in a variable 'result').
5. CATEGORICAL DATA: If columns are 'object' or 'category', and you are doing ML/Modeling, you MUST use 'ml.clean(df)' or 'ml.features(df)' to encode them first.
6. POWER TOOLS: You have access to the 'ml' (mlpilot) library. Use 'ml.analyze(df)', 'ml.baseline(df)', 'ml.explain(model, X)', etc.
7. FOR PLOTTING: Always use 'plt.figure(figsize=(10, 6))' before plotting. Use 'sns' or 'plt' functions. 
8. Return only the final answer or visual.

CODE:
"""
        return prompt

    def _get_llm_code(self, prompt: str) -> Optional[str]:
        if self.engine == "builtin":
            return self._call_builtin_llm(prompt)
            
        elif self.engine == "ollama":
            try:
                import ollama
                response = ollama.generate(model=self.model, prompt=prompt)
                code = response['response']
                return self._clean_code(code)
            except Exception as e:
                print_error(f"Ollama error: {e}")
                return None
        
        elif self.engine == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return self._clean_code(response.text)
            except Exception as e:
                if self.verbose:
                    print_error(f"Gemini error: {e}")
                # Fallback to builtin if cloud fails
                return self._call_builtin_llm(prompt)

        elif self.engine == "groq":
            try:
                import groq
                client = groq.Groq(api_key=self.groq_api_key)
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                code = completion.choices[0].message.content
                return self._clean_code(code)
            except Exception as e:
                print_error(f"Groq error: {e}")
                return None
        
        return None

    def _call_builtin_llm(self, prompt: str) -> Optional[str]:
        """Zero-Config local engine using shared engine."""
        code = call_builtin_llm(
            prompt, 
            system_prompt="You are a lead data scientist. Write Python code to solve the user's data request."
        )
        return self._clean_code(code) if code else None

    def _clean_code(self, code: str) -> str:
        # 1. Remove markdown ticks
        code = re.sub(r"```python\s?", "", code)
        code = re.sub(r"```\s?", "", code)
        
        # 2. Heuristic: Strip conversational tails (v1.2.2)
        # LLMs often add "This script calculates..." after the code.
        lines = code.split("\n")
        clean_lines = []
        for line in lines:
            low = line.lower()
            # If we hit conversational fluff, stop adding lines
            if any(marker in low for marker in ["this code", "this solution", "explanation:", "note:", "here is"]):
                if len(clean_lines) > 2: # Only break if we already have some code
                    break
            clean_lines.append(line)
        
        code = "\n".join(clean_lines).strip()
        
        # 3. Find the first line that looks like real code
        lines = code.split("\n")
        actual_code = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                # Remove common conversational prefixes (Labels)
                lower_s = stripped.lower()
                if lower_s.startswith("python:") or lower_s.startswith("code:") or lower_s.startswith("here:"):
                    continue
                
                # Check for code markers
                if re.match(r"^[a-zA-Z0-9_]+\s*=", stripped) or \
                   re.match(r"^(import |from |print|df|plt|sns|#|def |class )", stripped):
                    started = True
            
            if started:
                actual_code.append(line)
        
        cleaned = "\n".join(actual_code).strip()
        return cleaned if cleaned else code.strip()

    def _user_approval(self, code: str) -> bool:
        """Display code with Rich and ask for confirmation."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        
        console = Console()
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print("\n")
        console.print(Panel(syntax, title="AI Proposed Analysis", subtitle="[y] Run / [n] Cancel", border_style="cyan"))
        
        ans = input("  Run this code? (y/n): ").lower().strip()
        return ans == 'y'

    def _execute_code(self, code: str, attempts: int = 1) -> Any:
        # 1. Prepare Universal Workspace
        workspace = get_ds_workspace(df=self.df.copy())
        
        # FINAL BOSS HARDENING: Package Hijack (v1.2.2)
        # If the AI writes 'import ml', it will now work.
        import sys
        try:
            import mlpilot as ml
            sys.modules["ml"] = ml
            workspace["ml"] = ml
        except ImportError:
            pass
        
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        try:
            # 2. Execution (with Multi-Stage Recovery)
            with redirect_stdout(f):
                try:
                    try:
                        exec(code, {}, workspace)
                    except FileNotFoundError:
                        # STAGE 1: Hallucination Immunity (Strip file reads)
                        clean_code = self._strip_file_loading(code)
                        if clean_code != code:
                            exec(clean_code, {}, workspace)
                        else:
                            raise
                    except (ValueError, TypeError) as e:
                        # STAGE 2: Categorical Immunity (Auto-Encoding)
                        err_msg = str(e).lower()
                        if "could not convert string to float" in err_msg or "string" in err_msg:
                            if attempts > 0:
                                if self.verbose:
                                    print_step("Detecting categorical error. Attempting Autonomous Healing...")
                                
                                # Auto-Heal: Clean and encode the data in-place for the next run
                                try:
                                    import mlpilot as ml
                                    hardened_result = ml.features(workspace["df"], target=None, verbose=False)
                                    workspace["df"] = hardened_result.fit_transform(workspace["df"])
                                    
                                    # Retry execution with the same code but encoded data
                                    return self._execute_code(code, attempts=attempts-1)
                                except Exception:
                                    raise e
                            else:
                                raise e
                        else:
                            raise e
                except Exception:
                    # STAGE 3: Aggressive Recovery (AST-based Syntax Healing)
                    recovered_code = self._aggressive_recovery(code)
                    if recovered_code and recovered_code != code:
                        exec(recovered_code, {}, workspace)
                    else:
                        raise
        except Exception as e:
            print_error(f"Analysis failed during execution: {e}")
            if self.verbose:
                print(f"  FAILED CODE:\n{code}")
            return None
            
            output_captured = f.getvalue().strip()

            # 3. Intelligent Result Capture
            res = workspace.get("result")
            
            # If AI didn't explicitly set 'result', try to infer it from common vars
            if res is None:
                for candidate in ["df_result", "ans", "output", "summary"]:
                    if candidate in workspace:
                        res = workspace[candidate]
                        break
            
            # 4. Silent Professional Presentation
            if res is not None:
                print("")
                print_line()
                print(f"  ANSWER: {res}")
                print_line()
                print("")
            elif output_captured:
                # Capture anything the AI printed!
                print("")
                print_line()
                print(f"  ANALYSIS:\n{output_captured}")
                print_line()
                print("")
            else:
                # If no specific return variable found, at least confirm execution
                print("\n  [OK] Analysis executed successfully.")
            
            # 5. Visual Support (Auto-show plots if they were created)
            if workspace.get("plt") and len(workspace["plt"].get_fignums()) > 0:
                workspace["plt"].show()
            
            return res or output_captured
        except Exception as e:
            # PROFESSIONAL HARDENING: Don't be silent on error.
            print(f"\n  [ERROR] Analysis failed during execution: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _is_ollama_alive(self) -> bool:
        try:
            import ollama
            ollama.list()
            return True
        except Exception:
            return False

    def _detect_best_ollama_model(self) -> str:
        try:
            import ollama
            models = [m['name'] for m in ollama.list()['models']]
            priority = ["llama3:latest", "llama3", "mistral:latest", "mistral", "phi3"]
            for p in priority:
                if p in models:
                    return p
            return models[0] if models else "llama3"
        except Exception:
            return "llama3"


    def _aggressive_recovery(self, code: str) -> str:
        """
        ULTIMATE RECOVERY (v1.1.5):
        Use Python AST to scan the block and keep only valid statements.
        This handles multiline lists, dicts, and conversational tails.
        """
        import ast
        lines = code.split("\n")
        valid_lines = []
        
        # Strategy: Greedily add lines until a block is valid. 
        # Skip lines that are fundamentally not Python.
        current_block = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            
            # Simple heuristic to skip lines that are definitely just talking
            if not re.match(r"^[a-zA-Z0-9_\[\(#'\"].*", s):
                continue
                
            current_block.append(line)
            try:
                # Test the current accumulated block for syntax
                block_str = "\n".join(valid_lines + current_block)
                ast.parse(block_str)
                # If valid, merge current_block into valid_lines
                valid_lines.extend(current_block)
                current_block = []
            except SyntaxError:
                # If it's invalid, it might be an unfinished multiline block.
                # We keep it in current_block and try adding the NEXT line.
                pass
        
        return "\n".join(valid_lines).strip()

    def _strip_file_loading(self, code: str) -> str:
        """Surgically remove lines that try to load files (hallucinations)."""
        lines = code.split("\n")
        safe_lines = []
        for line in lines:
            s = line.strip()
            # If it tries to overwrite 'df' with a file load, skip it
            if "pd.read_csv" in s or "pd.read_excel" in s or "open(" in s:
                continue
            safe_lines.append(line)
        return "\n".join(safe_lines).strip()


    def _guided_setup_prompt(self):
        """Hidden by default in v0.2.0+. Only for manual debugging."""
        pass


def print_error(msg: str):
    from mlpilot.utils.display import print_error as pe
    pe(msg)
