import pandas as pd
import numpy as np
import dataclasses as dc

@dc.dataclass
class CleanReport:
    n_rows_before: int
    n_rows_after: int
    n_removed_non_daily: int
    n_removed_south: int
    n_removed_missing_target: int
    n_missing_after: int                      # <-- nieuw: totaal aantal missings (over alle cellen)
    dropped_by_reason: dict[str, int]         # <-- nieuw: breakdown per reden

def clean_dataframe(df_in: pd.DataFrame,
                    require_target: bool = True,
                    keep_only_region: str | None = "north",
                    keep_only_period: str | None = "daily") -> tuple[pd.DataFrame, CleanReport]:
    df = df_in.copy()
    n_before = len(df)
    dropped = {"non_daily": 0, "south": 0, "missing_target": 0}

    df.columns = [c.strip() for c in df.columns]

    # 1) Hou enkel daily
    before = len(df)
    if keep_only_period is not None and "period" in df.columns:
        df = df[df["period"].astype(str).str.lower() == keep_only_period.lower()]
    dropped["non_daily"] = before - len(df)

    # 2) Hou enkel north (Vlaanderen)
    before = len(df)
    if keep_only_region is not None and "reportType" in df.columns:
        df = df[df["reportType"].astype(str).str.lower() == keep_only_region.lower()]
    dropped["south"] = before - len(df)

    # 3) Target aanwezig (Kijkcijfers)
    if require_target and "Kijkcijfers" in df.columns:
        before = len(df)
        df["Kijkcijfers"] = pd.to_numeric(df["Kijkcijfers"], errors="coerce")
        df = df[df["Kijkcijfers"].notna()]
        dropped["missing_target"] = before - len(df)

    # 4) Types/conversies
    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce", dayfirst=True)
        df = df[df["Datum"].notna()]
    for col in ["Start","Duur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Dubbels weg
    df = df.drop_duplicates(subset=["Programma","Zender","Datum","Start","Duur"], keep="first")

    # Missings na cleaning (totaal aantal lege cellen)
    n_missing_after = int(df.isna().sum().sum())

    rep = CleanReport(
        n_rows_before = n_before,
        n_rows_after  = len(df),
        n_removed_non_daily = dropped["non_daily"],
        n_removed_south     = dropped["south"],
        n_removed_missing_target = dropped["missing_target"],
        n_missing_after = n_missing_after,
        dropped_by_reason = dropped,
    )
    return df, rep
