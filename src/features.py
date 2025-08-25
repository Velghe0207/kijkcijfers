from __future__ import annotations
import pandas as pd

# Kolommen die het model verwacht na feature build
CATEGORICAL = ["Programma","Zender","start_bin"]
NUMERIC     = ["Start","Duur","weekday","is_weekend","month","prime_time"]
FEATURE_ORDER = CATEGORICAL + NUMERIC  # consistent voor train/predict

def _start_bin_from_minutes(m: pd.Series) -> pd.Categorical:
    # nachtmorgenmiddagvooravondprime laat -> labels zonder spaties voor OneHot
    bins = [-1, 6*60, 12*60, 17*60, 20*60, 23*60, 24*60]
    labels = ["nacht","ochtend","middag","vooravond","prime","laat"]
    return pd.cut(m, bins=bins, labels=labels)

def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Voegt afgeleide features toe en levert FEATURE_ORDER kolommen + Datum terug."""
    df = df_in.copy()
    # Datum als datetime (kan al schoon zijn via cleaner)
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")

    # Kalender
    df["weekday"]    = df["Datum"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
    df["month"]      = df["Datum"].dt.month

    # Prime time
    start_fill = df["Start"].fillna(-1)
    df["prime_time"] = ((start_fill >= 20*60) & (start_fill < 23*60)).astype(int)

    # Startbin
    df["start_bin"] = _start_bin_from_minutes(df["Start"])

    # Zorg dat categorisch strings zijn
    df["Programma"] = df["Programma"].astype(str)
    df["Zender"]    = df["Zender"].astype(str)

    # Output: bewaar ook Datum voor tijdssplit, maar features in vaste volgorde
    out = df.copy()
    # maak zeker dat alle kolommen bestaan
    for c in FEATURE_ORDER:
        if c not in out.columns:
            out[c] = pd.NA
    return out[["Datum"] + FEATURE_ORDER]
