"""
mlpilot/eda/report.py
EDA HTML report builder. Inherits from HTMLReportBuilder.
Assembles the 12-section EDA report from an EDAResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from mlpilot.utils.report_base import HTMLReportBuilder

if TYPE_CHECKING:
    from mlpilot.eda.analyzer import EDAResult


class EDAReportBuilder(HTMLReportBuilder):
    """Builds the full EDA HTML report from an EDAResult object."""

    def __init__(self, result: "EDAResult"):
        super().__init__(
            title="Exploratory Data Analysis",
            subtitle=f"Dataset: {result.summary.n_rows:,} rows × {result.summary.n_cols} columns",
        )
        self.result = result

    def build_full_report(self) -> "EDAReportBuilder":
        """Build all sections. Call .build() or .save() afterwards."""
        r = self.result
        self._add_overview(r)
        self._add_quality(r)
        self._add_missing(r)
        self._add_distributions(r)
        self._add_categorical(r)
        self._add_correlations(r)
        if r.target_analysis is not None:
            self._add_target(r)
        self._add_outliers(r)
        self._add_issues(r)
        self._add_recommendations(r)
        return self

    # ------------------------------------------------------------------

    def _add_overview(self, r: "EDAResult") -> None:
        s = r.summary
        stats = [
            ("Rows", f"{s.n_rows:,}", ""),
            ("Columns", s.n_cols, ""),
            ("Numeric", s.n_numeric, ""),
            ("Categorical", s.n_categorical, ""),
            ("Missing Cells", f"{s.n_missing_cells:,}", ""),
            ("Missing %", f"{s.pct_missing_cells:.1f}%", ""),
            ("Duplicates", f"{s.n_duplicate_rows:,}", ""),
            ("Memory", f"{s.memory_mb:.1f} MB", ""),
        ]
        content = self.stat_grid(stats, cols=4)
        # dtype breakdown table
        dtype_rows = [
            ["Numeric", str(s.n_numeric)],
            ["Categorical", str(s.n_categorical)],
            ["DateTime", str(s.n_datetime)],
            ["Text", str(s.n_text)],
            ["Boolean", str(s.n_boolean)],
        ]
        content += self.table(["Data Type", "Count"], dtype_rows)
        self.add_section(self.section("📊 Dataset Overview", content, ""))

    def _add_quality(self, r: "EDAResult") -> None:
        score = r.quality_score
        bar = self.score_bar(score)
        badge = self.score_badge(score)
        desc = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Attention"
        content = f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
          <div class="stat-value" style="font-size:3rem">{score:.0f}</div>
          <div>
            <div style="font-size:1.2rem;font-weight:600">{desc}</div>
            <div style="color:var(--muted);font-size:0.85rem">Data Quality Score (out of 100)</div>
          </div>
          {badge}
        </div>
        {bar}
        """
        self.add_section(self.section("🏆 Data Quality Score", content))

    def _add_missing(self, r: "EDAResult") -> None:
        missing_cols = [
            (col, p) for col, p in
            ((col, prof.pct_missing) for col, prof in r.column_profiles.items())
            if p > 0
        ]
        if not missing_cols:
            content = '<p style="color:var(--success)">✓ No missing values found</p>'
        else:
            rows = sorted(missing_cols, key=lambda x: -x[1])
            tbl_rows = [[col, f"{pct:.1f}%",
                         "🔴" if pct > 50 else "🟡" if pct > 20 else "🟢"]
                        for col, pct in rows]
            content = self.table(["Column", "% Missing", "Severity"], tbl_rows)

            # Embed missing heatmap if plotly is available
            if r.plots.get("missing_heatmap"):
                content += self.embed_plotly(r.plots["missing_heatmap"], height=300)

        self.add_section(self.section("🕳️ Missing Values", content))

    def _add_distributions(self, r: "EDAResult") -> None:
        numeric_profiles = [
            (col, prof) for col, prof in r.column_profiles.items()
            if prof.is_numeric
        ]
        if not numeric_profiles:
            self.add_section(self.section("📈 Distributions", "<p>No numeric columns.</p>"))
            return

        rows = []
        for col, prof in numeric_profiles:
            rows.append([
                col,
                f"{prof.mean:.3g}" if prof.mean is not None else "—",
                f"{prof.median:.3g}" if prof.median is not None else "—",
                f"{prof.std:.3g}" if prof.std is not None else "—",
                f"{prof.skewness:.2f}" if prof.skewness is not None else "—",
                f"{prof.n_outliers_iqr}",
            ])
        content = self.table(
            ["Column", "Mean", "Median", "Std", "Skewness", "IQR Outliers"], rows
        )

        # Embed sample histograms (up to 6)
        hist_html = '<div class="grid-2" style="margin-top:1rem">'
        embedded = 0
        for col, _ in numeric_profiles[:6]:
            fig = r.plots.get(f"hist_{col}")
            if fig is not None:
                hist_html += f'<div>{self.embed_plotly(fig, height=280)}</div>'
                embedded += 1
        hist_html += "</div>"
        if embedded:
            content += hist_html

        self.add_section(self.section("📈 Numeric Distributions", content))

    def _add_categorical(self, r: "EDAResult") -> None:
        cat_profiles = [
            (col, prof) for col, prof in r.column_profiles.items()
            if prof.is_categorical
        ]
        if not cat_profiles:
            self.add_section(self.section("🏷️ Categorical Columns", "<p>No categorical columns.</p>"))
            return
        rows = []
        for col, prof in cat_profiles:
            top = prof.top_values[0][0] if prof.top_values else "—"
            top_cnt = prof.top_values[0][1] if prof.top_values else 0
            rows.append([col, str(prof.n_unique), prof.cardinality, str(top), str(top_cnt)])
        content = self.table(
            ["Column", "Unique Values", "Cardinality", "Top Value", "Count"], rows
        )

        bar_html = '<div class="grid-2" style="margin-top:1rem">'
        embedded = 0
        for col, _ in cat_profiles[:4]:
            fig = r.plots.get(f"bar_{col}")
            if fig is not None:
                bar_html += f'<div>{self.embed_plotly(fig, height=280)}</div>'
                embedded += 1
        bar_html += "</div>"
        if embedded:
            content += bar_html

        self.add_section(self.section("🏷️ Categorical Columns", content))

    def _add_correlations(self, r: "EDAResult") -> None:
        if r.correlations is None:
            self.add_section(self.section("🔗 Correlations", "<p>Not computed.</p>"))
            return

        high_corr = r.correlations.high_pairs
        if high_corr:
            rows = [[a, b, f"{v:.3f}"] for a, b, v in high_corr]
            content = f"<p style='color:var(--warning)'>⚠ {len(high_corr)} highly correlated pair(s) found (|r| > threshold):</p>"
            content += self.table(["Feature A", "Feature B", "Correlation"], rows)
        else:
            content = '<p style="color:var(--success)">✓ No high-correlation pairs above threshold</p>'

        heatmap_fig = r.plots.get("correlation_heatmap")
        if heatmap_fig is not None:
            content += self.embed_plotly(heatmap_fig, height=500)

        self.add_section(self.section("🔗 Correlations", content))

    def _add_target(self, r: "EDAResult") -> None:
        ta = r.target_analysis
        severity_color = {
            "none": "var(--success)", "mild": "var(--accent)",
            "moderate": "var(--warning)", "severe": "var(--danger)", "critical": "var(--danger)"
        }
        color = severity_color.get(ta.imbalance_severity, "var(--text)")
        content = f"""
        <div class="grid-3">
          <div class="stat-box">
            <div class="stat-value">{ta.imbalance_score:.0f}/100</div>
            <div class="stat-label">Imbalance Score</div>
          </div>
          <div class="stat-box">
            <div class="stat-value" style="color:{color}">{ta.imbalance_severity.title()}</div>
            <div class="stat-label">Imbalance Severity</div>
          </div>
          <div class="stat-box">
            <div class="stat-value" style="font-size:1rem">{ta.recommended_strategy}</div>
            <div class="stat-label">Recommended Strategy</div>
          </div>
        </div>
        """
        # Class distribution table
        dist_rows = [[str(k), str(v), f"{v/sum(ta.class_distribution.values())*100:.1f}%"]
                     for k, v in ta.class_distribution.items()]
        content += self.table(["Class", "Count", "%"], dist_rows)

        # Target distribution plot
        fig = r.plots.get("target_distribution")
        if fig is not None:
            content += self.embed_plotly(fig, height=300)

        # Top correlated features
        if ta.top_correlated:
            corr_rows = [[col, f"{val:.3f}"] for col, val in ta.top_correlated]
            content += "<h3 style='margin-top:1rem'>Top Features Correlated with Target</h3>"
            content += self.table(["Feature", "Correlation"], corr_rows)

        self.add_section(self.section("🎯 Target Analysis", content))

    def _add_outliers(self, r: "EDAResult") -> None:
        outlier_cols = [
            (col, prof.n_outliers_iqr, prof.n_outliers_zscore)
            for col, prof in r.column_profiles.items()
            if prof.is_numeric and (prof.n_outliers_iqr > 0 or prof.n_outliers_zscore > 0)
        ]
        if not outlier_cols:
            content = '<p style="color:var(--success)">✓ No significant outliers detected</p>'
        else:
            rows = [[col, str(iqr), str(z)] for col, iqr, z in outlier_cols]
            content = self.table(["Column", "IQR Outliers", "Z-Score Outliers"], rows)
        self.add_section(self.section("⚠️ Outliers", content))

    def _add_issues(self, r: "EDAResult") -> None:
        content = self.issue_list(r.issues)
        self.add_section(self.section("🚨 Issues Detected", content))

    def _add_recommendations(self, r: "EDAResult") -> None:
        if not r.recommendations:
            content = '<p style="color:var(--success)">✓ No recommendations</p>'
        else:
            rows = [
                [str(rec.priority), rec.action, rec.column or "—", rec.reason]
                for rec in sorted(r.recommendations, key=lambda x: x.priority)
            ]
            content = self.table(["Priority", "Action", "Column", "Reason"], rows,
                                 highlight_col=1)
            # Code snippets
            snippets = [rec.suggested_code for rec in r.recommendations
                        if rec.suggested_code]
            if snippets:
                content += "<h3 style='margin-top:1rem'>Suggested Code</h3>"
                content += self.code_block("\n".join(snippets))
        self.add_section(self.section("💡 Recommendations", content))
