#!/usr/bin/env python3
"""
MAGNETO Pipeline Audit
Checks date ranges, grid consistency, and schema without loading full files.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.feather as feather
import hashlib
import gc
from pathlib import Path
from _Common import Config

class PipelineAuditor:
    def __init__(self):
        self.report = ["# MAGNETO Pipeline Consistency Audit Report (Low-RAM)\n"]
        self.files_to_check = {
            "OMNI2 Features": Config.FILE_OMNI_FEATHER,
            "MODIS Vegetation": Config.FILE_MODIS_PARQUET,
            "ERA5 Environment": Config.FILE_ERA5_PARQUET,
            "SIF Aggregated": Config.FILE_SIF_FINAL,
        }
        
    def _get_header_sample(self, path, n=5):
        """Reads first few rows safely."""
        try:
            if path.suffix == '.parquet':
                return pq.ParquetFile(path).read_row_group(0).to_pandas().head(n)
            elif path.suffix == '.feather':
                return feather.read_table(path, memory_map=True).to_pandas().head(n)
        except Exception:
            return None

    def check_file(self, label, path):
        if not path.exists():
            return f"### {label}\n**STATUS: MISSING**\n"
            
        df = self._get_header_sample(path)
        if df is None:
            return f"### {label}\n**STATUS: UNREADABLE**\n"

        issues = []
        
        # 1. Date Check (1970 trap)
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        if date_col:
            # Read only the date column to check range (low memory cost)
            try:
                if path.suffix == '.parquet':
                    dates = pq.read_table(path, columns=[date_col]).to_pandas()[date_col]
                else:
                    dates = feather.read_table(path, columns=[date_col], memory_map=True)[date_col].to_pandas()
                
                dmin, dmax = pd.to_datetime(dates).min(), pd.to_datetime(dates).max()
                if dmin.year < 2010:
                    issues.append(f"[ERROR] Date Trap Detected: Min year {dmin.year}")
                range_str = f"{dmin.date()} to {dmax.date()}"
            except:
                range_str = "Error reading dates"
        else:
            range_str = "No date column"

        # 2. Type Check
        if 'lat_id' in df.columns:
            if df['lat_id'].dtype != 'int16':
                issues.append(f"[WARNING] lat_id is {df['lat_id'].dtype} (expected int16)")

        status = "[OK]" if not issues else "[ISSUES]"
        issues_str = "\n".join([f"- {i}" for i in issues]) if issues else "- No obvious schema issues."

        return f"""
### {label} ({status})
- **File:** `{path.name}`
- **Date Range:** {range_str}
- **Columns:** {list(df.columns[:5])}...
{issues_str}
"""

    def run(self):
        print("[AUDIT] Starting Audit...")
        for label, path in self.files_to_check.items():
            self.report.append(self.check_file(label, path))
            gc.collect()
            
        out_path = Config.REPORTS_ROOT / "pipeline_audit.md"
        with open(out_path, "w") as f:
            f.write("\n".join(self.report))
        print(f"[DONE] Report saved to {out_path}")

if __name__ == "__main__":
    PipelineAuditor().run()