"""
mlpilot/utils/report_base.py
Base HTML report builder — no Jinja2 dependency, pure Python string templating.
All module-specific report builders inherit from this.
"""

from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_CSS = """
:root {
  --bg: #0f111a;
  --card: #1a1d2e;
  --accent: #6c63ff;
  --accent2: #00d9c6;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --border: #2d3150;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--text);
  padding: 2rem;
  line-height: 1.6;
}
h1, h2, h3 { margin-bottom: 0.75rem; }
h1 { font-size: 2rem; background: linear-gradient(135deg, var(--accent), var(--accent2));
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.4rem; color: var(--accent); border-bottom: 1px solid var(--border);
     padding-bottom: 0.5rem; margin-top: 2rem; }
h3 { font-size: 1.1rem; color: var(--accent2); }
.header { display: flex; justify-content: space-between; align-items: center;
          border-bottom: 2px solid var(--accent); padding-bottom: 1rem; margin-bottom: 2rem; }
.badge { background: var(--accent); color: white; padding: 0.25rem 0.75rem;
         border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.badge-success { background: var(--success); }
.badge-warning { background: var(--warning); color: #1a1a1a; }
.badge-danger  { background: var(--danger); }
.badge-info    { background: var(--accent2); color: #1a1a1a; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1.5rem; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
.grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; }
.stat-box { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
            padding: 1rem; text-align: center; }
.stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.stat-label { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th { background: #252840; color: var(--accent); text-align: left;
     padding: 0.75rem 1rem; font-size: 0.85rem; }
td { padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.875rem; }
tr:hover td { background: #1e2035; }
.score-bar { height: 8px; border-radius: 4px; background: var(--border); }
.score-fill { height: 100%; border-radius: 4px;
              background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.issue-critical { border-left: 3px solid var(--danger); }
.issue-warning  { border-left: 3px solid var(--warning); }
.issue-info     { border-left: 3px solid var(--accent); }
.issue { padding: 0.75rem 1rem; margin: 0.5rem 0; background: #1c1f35;
         border-radius: 0 6px 6px 0; }
code { background: #252840; padding: 0.1rem 0.4rem; border-radius: 4px;
       font-family: 'Fira Code', monospace; font-size: 0.85rem; color: var(--accent2); }
pre { background: #1a1d2e; border: 1px solid var(--border); border-radius: 8px;
      padding: 1rem; overflow-x: auto; }
pre code { background: none; padding: 0; }
.tag { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px;
       font-size: 0.75rem; font-weight: 600; margin: 0.1rem; }
.tag-missing { background: #3b1f1f; color: #f87171; }
.tag-outlier { background: #3b2f1a; color: #fbbf24; }
.tag-cardinality { background: #1a2b3b; color: #60a5fa; }
"""

_JS = """
document.addEventListener('DOMContentLoaded', () => {
  // Animate stat values
  document.querySelectorAll('.stat-value[data-value]').forEach(el => {
    const target = parseFloat(el.dataset.value);
    const isFloat = el.dataset.decimals !== undefined;
    const decimals = parseInt(el.dataset.decimals || '0');
    let start = 0;
    const step = target / 40;
    const interval = setInterval(() => {
      start = Math.min(start + step, target);
      el.textContent = isFloat ? start.toFixed(decimals) : Math.floor(start).toLocaleString();
      if (start >= target) clearInterval(interval);
    }, 20);
  });
});
"""


class HTMLReportBuilder:
    """
    Base HTML report builder.
    Usage:
        builder = HTMLReportBuilder(title='EDA Report')
        builder.add_section(builder.stat_grid([...]))
        html = builder.build()
        builder.save('report.html')
    """

    def __init__(self, title: str = "mlpilot Report", subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self._sections: List[str] = []
        self._generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Section building blocks
    # ------------------------------------------------------------------

    def add_section(self, html: str) -> "HTMLReportBuilder":
        self._sections.append(html)
        return self

    def stat_grid(self, stats: List[Tuple[str, Any, str]], cols: int = 4) -> str:
        """stats = [(label, value, badge_class), ...]"""
        items = []
        for label, value, badge_cls in stats:
            val_str = str(value)
            items.append(f"""
            <div class="stat-box">
              <div class="stat-value">{val_str}</div>
              <div class="stat-label">{label}</div>
            </div>""")
        return f'<div class="grid-{min(cols, 4)}" style="margin:1rem 0">{"".join(items)}</div>'

    def section(self, title: str, content: str, icon: str = "") -> str:
        return f"""
        <div class="card">
          <h2>{icon} {title}</h2>
          {content}
        </div>"""

    def table(self, headers: List[str], rows: List[List[str]],
              highlight_col: Optional[int] = None) -> str:
        th_html = "".join(f"<th>{h}</th>" for h in headers)
        rows_html = []
        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                style = ' style="color:var(--accent2);font-weight:600"' if i == highlight_col else ""
                cells.append(f"<td{style}>{cell}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        return f"""<table><thead><tr>{th_html}</tr></thead><tbody>{"".join(rows_html)}</tbody></table>"""

    def score_badge(self, score: float) -> str:
        if score >= 80:
            cls = "badge-success"
        elif score >= 60:
            cls = "badge-warning"
        else:
            cls = "badge-danger"
        return f'<span class="badge {cls}">Quality: {score:.0f}/100</span>'

    def score_bar(self, score: float) -> str:
        color = "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
        return f"""
        <div class="score-bar">
          <div class="score-fill" style="width:{score:.0f}%;background:{color}"></div>
        </div>"""

    def embed_plotly(self, fig, height: int = 400) -> str:
        """Embed a plotly figure as interactive HTML."""
        try:
            import plotly.io as pio
            html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                               config={"displayModeBar": False})
            return f'<div style="height:{height}px">{html}</div>'
        except Exception:
            return "<p><em>Plot unavailable (plotly not installed)</em></p>"

    def embed_matplotlib(self, fig) -> str:
        """Embed a matplotlib figure as a base64 PNG."""
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                        facecolor="#1a1d2e")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px"/>'
        except Exception:
            return "<p><em>Plot unavailable</em></p>"

    def issue_list(self, issues: Any) -> str:
        if not issues:
            return '<p style="color:var(--success)">✓ No issues found</p>'
        items = []
        for issue in issues:
            severity = getattr(issue, "severity", "info")
            message = getattr(issue, "message", str(issue))
            col = getattr(issue, "column", None)
            loc = f"<code>{col}</code> — " if col else ""
            items.append(f'<div class="issue issue-{severity}">{loc}{message}</div>')
        return "".join(items)

    def code_block(self, code: str, language: str = "python") -> str:
        return f"<pre><code class='language-{language}'>{code}</code></pre>"

    # ------------------------------------------------------------------
    # Final HTML assembly
    # ------------------------------------------------------------------

    def build(self) -> str:
        body = "\n".join(self._sections)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title} — mlpilot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code&display=swap" rel="stylesheet">
  <style>{_CSS}</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🚀 {self.title}</h1>
      <p style="color:var(--muted)">{self.subtitle}</p>
    </div>
    <div>
      <span class="badge">mlpilot</span>
      <span style="color:var(--muted);font-size:0.8rem;margin-left:1rem">
        Generated: {self._generated_at}
      </span>
    </div>
  </div>
  {body}
  <footer style="text-align:center;color:var(--muted);font-size:0.8rem;margin-top:3rem;
                 border-top:1px solid var(--border);padding-top:1rem">
    Generated by <strong>mlpilot</strong> • The Complete Python ML Library
  </footer>
  <script>{_JS}</script>
</body>
</html>"""

    def save(self, path: str) -> str:
        """Save the report to disk. Creates parent directories. Returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self.build()
        out.write_text(html, encoding="utf-8")
        return str(out.resolve())
