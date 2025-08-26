# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple
# from pathlib import Path
# import pandas as pd
# import numpy as np

# REQUIRED_MIN_COLS = ["Programma","Zender","Datum","Start","Duur"]
# TARGET_COL = "Kijkcijfers"
# MAX_MINUTES_PER_DAY = 24*60
# MAX_DURATION_MIN = 5*60
# MIN_DURATION_MIN = 1
# MIN_VIEWERS = 1

# @dataclass
# class CleanReport:
#     n_rows_before: int
#     n_rows_after: int
#     dropped_by_reason: Dict[str, int]
#     n_missing_after: Dict[str, int]

# def _as_minutes(x) -> Optional[float]:
#     if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
#     s = str(x).strip()
#     if s == "": return np.nan
#     try: return float(s)
#     except Exception: pass
#     parts = s.split(":")
#     if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
#         h, m = int(parts[0]), int(parts[1])
#         mins = h*60 + m
#         if len(parts) == 3 and parts[2].isdigit():
#             if int(parts[2]) >= 30: mins += 1
#         return float(mins)
#     return np.nan

# def validate_schema(df: pd.DataFrame, require_target: bool = False) -> Tuple[bool, List[str]]:
#     required = REQUIRED_MIN_COLS + ([TARGET_COL] if require_target else [])
#     missing = [c for c in required if c not in df.columns]
#     return (len(missing) == 0, missing)

# def clean_dataframe(
#     df: pd.DataFrame,
#     require_target: bool = False,
#     keep_only_region: Optional[str] = "north",
#     drop_duplicates: bool = True
# ) -> Tuple[pd.DataFrame, CleanReport]:
#     df = df.copy()
#     n_before = len(df)
#     dropped: Dict[str, int] = {}

#     if keep_only_region and "reportType" in df.columns:
#         mask = (df["reportType"] == keep_only_region)
#         dropped[f"reportType != {keep_only_region}"] = int((~mask).sum())
#         df = df[mask].copy()

#     ok, missing = validate_schema(df, require_target=require_target)
#     if not ok: raise ValueError(f"Ontbrekende kolommen: {missing}")

#     for c in ["Programma","Zender"]:
#         df[c] = df[c].astype(str).str.strip()

#     df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
#     bad = df["Datum"].isna().sum()
#     if bad:
#         dropped["invalid Datum"] = int(bad)
#         df = df.dropna(subset=["Datum"])

#     df["Start"] = df["Start"].apply(_as_minutes)
#     df["Duur"]  = df["Duur"].apply(_as_minutes)

#     mask_prog = df["Programma"].str.len() > 0
#     dropped["empty Programma"] = int((~mask_prog).sum())
#     df = df[mask_prog]

#     mask_zend = df["Zender"].str.len() > 0
#     dropped["empty Zender"] = int((~mask_zend).sum())
#     df = df[mask_zend]

#     mask_start = df["Start"].between(0, MAX_MINUTES_PER_DAY, inclusive="both")
#     dropped["invalid Start"] = int((~mask_start).sum())
#     df = df[mask_start]

#     mask_duur = df["Duur"].between(MIN_DURATION_MIN, MAX_DURATION_MIN, inclusive="both")
#     dropped["invalid Duur"] = int((~mask_duur).sum())
#     df = df[mask_duur]

#     if require_target:
#         df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
#         mask_target = df[TARGET_COL].notna() & (df[TARGET_COL] >= MIN_VIEWERS)
#         dropped["invalid Kijkcijfers"] = int((~mask_target).sum())
#         df = df[mask_target]

#     if drop_duplicates:
#         n_dup = int(df.duplicated(subset=["Programma","Zender","Datum"]).sum())
#         if n_dup: dropped["duplicates"] = n_dup
#         df = df.drop_duplicates(subset=["Programma","Zender","Datum"], keep="first")

#     df["Start"] = df["Start"].round(0).astype("Int64")
#     df["Duur"]  = df["Duur"].round(0).astype("Int64")
#     if require_target:
#         df[TARGET_COL] = df[TARGET_COL].round(0).astype("Int64")

#     report = CleanReport(
#         n_rows_before=n_before,
#         n_rows_after=len(df),
#         dropped_by_reason={k:v for k,v in dropped.items() if v},
#         n_missing_after=df.isna().sum().to_dict()
#     )
#     return df, report

# def clean_file_to_csv(input_path: str|Path, output_path: str|Path, require_target: bool=False, keep_only_region: Optional[str]="north") -> CleanReport:
#     df = pd.read_csv(input_path, sep=";")
#     df_clean, rep = clean_dataframe(df, require_target=require_target, keep_only_region=keep_only_region)
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#     df_clean.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
#     return rep

# src/clean.py
from __future__ import annotations
import dataclasses as dc
import pandas as pd

@dc.dataclass
class CleanReport:
    n_rows_before: int
    n_rows_after: int
    n_removed_non_daily: int
    n_removed_south: int
    n_removed_missing_target: int

def clean_dataframe(df_in: pd.DataFrame,
                    require_target: bool = True,
                    keep_only_region: str | None = "north",
                    keep_only_period: str | None = "daily") -> tuple[pd.DataFrame, CleanReport]:
    df = df_in.copy()
    n_before = len(df)

    # Normaliseer kolomnamen (zeker als uit verschillende bronnen)
    df.columns = [c.strip() for c in df.columns]

    # --- 1) Hou alleen 'daily' records
    n_non_daily_before = len(df)
    if keep_only_period is not None and "period" in df.columns:
        df = df[df["period"].astype(str).str.lower() == keep_only_period.lower()]
    n_removed_non_daily = n_non_daily_before - len(df)

    # --- 2) Hou alleen Vlaanderen/noord
    n_south_before = len(df)
    if keep_only_region is not None and "reportType" in df.columns:
        df = df[df["reportType"].astype(str).str.lower() == keep_only_region.lower()]
    n_removed_south = n_south_before - len(df)

    # --- 3) Target aanwezig?
    n_missing_target = 0
    if require_target:
        # target kolom heet bij jou 'Kijkcijfers'
        if "Kijkcijfers" in df.columns:
            n_target_before = len(df)
            df = df[pd.to_numeric(df["Kijkcijfers"], errors="coerce").notna()]
            n_missing_target = n_target_before - len(df)
            df["Kijkcijfers"] = pd.to_numeric(df["Kijkcijfers"], errors="coerce")
        else:
            # Als target ontbreekt (bv. inference), doen we niets
            pass

    # --- 4) Basis conversies
    # Datum
    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce", dayfirst=True)
        df = df[df["Datum"].notna()]
    # Start en Duur (minuten)
    for col in ["Start", "Duur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 5) Dubbele/rare rijen optioneel verwijderen (kort)
    df = df.drop_duplicates(subset=["Programma", "Zender", "Datum", "Start", "Duur"], keep="first")

    rep = CleanReport(
        n_rows_before=n_before,
        n_rows_after=len(df),
        n_removed_non_daily=n_removed_non_daily,
        n_removed_south=n_removed_south,
        n_removed_missing_target=n_missing_target,
    )
    return df, rep
