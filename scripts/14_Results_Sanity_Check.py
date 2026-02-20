#!/usr/bin/env python3
"""
Results Sanity Checker for MAGNETO Pipeline.
Validates ALL pipeline artifacts: Spearman screening, Matrix Search, and Meta-analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from _Common import Config

class ResultsSanityChecker:
    def __init__(self):
        self.report = ["# MAGNETO Pipeline Results Audit\n"]
        self.results_dir = Config.RESULTS_DIR
        self.meta_dir = Config.META_ANALYSIS_DIR
        
    def _validate_csv(self, path, min_rows=1, required_cols=None):
        """
        Deep inspection of a CSV file.
        Returns: (Status Symbol, Details String)
        """
        if not path.exists():
            return "[MISSING]", "**MISSING**"
        
        if path.stat().st_size == 0:
            return "[EMPTY]", "**EMPTY FILE**"

        try:
            # Read file
            df = pd.read_csv(path)
            rows, cols = df.shape
            
            # 1. Check for emptiness
            if rows < min_rows:
                return "[LOW DATA]", f"**LOW DATA** ({rows} rows < {min_rows})"
            
            # 2. Check columns
            if required_cols:
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    return "[SCHEMA ERROR]", f"**SCHEMA ERROR** (Missing: {missing})"
            
            # 3. Check for NaNs in critical columns
            # Check t_sii for Matrix Search
            if 't_sii' in df.columns:
                valid_t = df['t_sii'].notna().sum()
                if valid_t == 0:
                    return "[ERROR]", "**ALL NaNs** in `t_sii`"
                if valid_t < rows:
                    return "[OK]", f"OK ({rows} rows, {rows-valid_t} NaNs)"
            
            # Check spearman_r for Spearman analysis
            if 'spearman_r' in df.columns:
                valid_r = df['spearman_r'].notna().sum()
                if valid_r == 0:
                    return "[ERROR]", "**ALL NaNs** in `spearman_r`"

            return "[OK]", f"OK ({rows} rows)"

        except Exception as e:
            return "[ERROR]", f"**READ ERROR**: {str(e)}"

    def check_step10_spearman(self):
        """Validates Step 10 Output (Mechanism & Screening)."""
        self.report.append("## 1. Spearman Screening & Mechanism (Step 10)")
        
        files = [
            # Filename, Min rows, Columns
            ("spearman_mechanism_SII_vs_ENV.csv", 10, ['parameter_1', 'spearman_r', 'p_value']),
            ("screening_overview_global.csv", 4, ['target', 'spearman_r', 'p_value'])
        ]
        
        for fname, min_r, cols in files:
            fpath = self.results_dir / fname
            icon, msg = self._validate_csv(fpath, min_rows=min_r, required_cols=cols)
            self.report.append(f"- {icon} **`{fname}`**: {msg}")

    def check_step11_matrix(self):
        """Validates Step 11 Output (Matrix Search GPU)."""
        self.report.append("\n## 2. Matrix Search Results (Step 11)")
        self.report.append("| Target | Scenario | Status | Details |")
        self.report.append("|---|---|---|---|")
        
        targets = Config.SPEARMAN_TARGETS
        scenarios = Config.SCENARIO_MASKS.keys()
        
        # Expected columns in regression results
        req_cols = ['bin_label', 't_sii', 'p_sii', 'n_eff']
        
        total_found = 0
        total_expected = len(targets) * len(scenarios)

        for t in targets:
            for s in scenarios:
                fname = f"matrix_search_{t}_{s}.csv"
                fpath = self.results_dir / fname
                
                # Expect >1000 models (reality is >100k)
                icon, msg = self._validate_csv(fpath, min_rows=1000, required_cols=req_cols)
                
                if "OK" in msg:
                    total_found += 1
                
                self.report.append(f"| `{t}` | `{s}` | {icon} | {msg} |")
        
        summary_icon = "[OK]" if total_found == total_expected else "[WARNING]"
        self.report.append(f"\n**Summary:** {summary_icon} Valid Files: {total_found}/{total_expected}")

    def check_step12_meta(self):
        """Validates Step 12 Output (Meta-Analysis)."""
        self.report.append("\n## 3. Meta-Statistics (Step 12)")
        
        files = [
            ("meta_summary_120rows.csv", 16, ['metric', 'mask', 'bin', 't_sii']),
            ("meta_summary_60combos_wide.csv", 16, ['metric', 'mask', 't_sii_MA'])
        ]
        
        for fname, min_r, cols in files:
            fpath = self.meta_dir / fname
            icon, msg = self._validate_csv(fpath, min_rows=min_r, required_cols=cols)
            self.report.append(f"- {icon} **`{fname}`**: {msg}")

    def run(self):
        print("[CHECK] Starting Full Pipeline Audit...")
        
        self.check_step10_spearman()
        self.check_step11_matrix()
        self.check_step12_meta()
        
        # Save report
        out_path = Config.REPORTS_ROOT / "results_sanity_check.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w") as f:
            f.write("\n".join(self.report))
            
        print(f"[SUCCESS] Audit completed. Report saved to: {out_path}")
        print("\n" + "\n".join(self.report))

if __name__ == "__main__":
    ResultsSanityChecker().run()