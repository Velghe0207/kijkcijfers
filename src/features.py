# from __future__ import annotations
# from pathlib import Path
# import pandas as pd
# import numpy as np

# # === Pad naar lokale weerdata ===
# WEATHER_PATH = Path("data/processed/weather_daily.csv")

# # === Basis features ===
# CATEGORICAL = ["Programma", "Zender", "start_bin"]
# NUMERIC_BASE = ["Start", "Duur", "weekday", "is_weekend", "month", "prime_time"]

# # === Weer features (compact & informatief) ===
# WEATHER_NUMERIC = [
#     "rain_mm",          # uit rain_sum
#     "wind_max_kmh",     # uit wind_speed_10m_max
#     "temp_max_c",       # uit temperature_2m_max
#     "temp_min_c",       # uit temperature_2m_min
#     "sunshine_h",       # uit sunshine_duration / 3600
#     "uv_index_max",     # uv index
#     "temp_range_c",     # max - min
# ]
# WEATHER_BINARY = [
#     "is_rainy",         # rain_mm >= 1.0
#     "is_windy",         # wind_max_kmh >= 50
#     "is_hot",           # temp_max_c >= 25
#     "is_cold",          # temp_min_c <= 0
# ]

# NUMERIC = NUMERIC_BASE + WEATHER_NUMERIC + WEATHER_BINARY
# NUMERIC     = ["Start","Duur","weekday","is_weekend","month","prime_time"]

# FEATURE_ORDER = CATEGORICAL + NUMERIC  # consistente volgorde voor train/predict


# def _start_bin_from_minutes(m: pd.Series) -> pd.Categorical:
#     bins   = [-1, 6*60, 12*60, 17*60, 20*60, 23*60, 24*60]
#     labels = ["nacht", "ochtend", "middag", "vooravond", "prime", "laat"]
#     return pd.cut(m, bins=bins, labels=labels)

# def _load_weather(weather_path: Path = WEATHER_PATH) -> pd.DataFrame | None:
#     """Lees weerdata en prepareer kolommen voor merge op Datum (dag-niveau)."""
#     if not weather_path.exists():
#         return None
#     w = pd.read_csv(weather_path)

#     # 'date' → lokale dag (Europe/Brussels) en maak 'Datum'
#     date_raw = pd.to_datetime(w["date"], errors="coerce", utc=True)  # werkt voor UTC en naïef
#     date_local = date_raw.dt.tz_convert("Europe/Brussels")
#     w["Datum"] = pd.to_datetime(date_local.dt.date)  # dag-niveau (tz weg)

#     # Normaliseren / renames
#     w["sunshine_h"] = w.get("sunshine_duration", 0) / 3600.0
#     w = w.rename(columns={
#         "rain_sum": "rain_mm",
#         "wind_speed_10m_max": "wind_max_kmh",
#         "temperature_2m_max": "temp_max_c",
#         "temperature_2m_min": "temp_min_c",
#     })
#     # Afgeleiden
#     w["temp_range_c"] = w["temp_max_c"] - w["temp_min_c"]
#     w["is_rainy"] = (w["rain_mm"].fillna(0) >= 1.0).astype(int)
#     w["is_windy"] = (w["wind_max_kmh"].fillna(0) >= 50).astype(int)
#     w["is_hot"]   = (w["temp_max_c"].fillna(-999) >= 25).astype(int)
#     w["is_cold"]  = (w["temp_min_c"].fillna( 999) <= 0).astype(int)

#     keep = ["Datum", "uv_index_max", "sunshine_h", "rain_mm",
#             "wind_max_kmh", "temp_max_c", "temp_min_c", "temp_range_c",
#             "is_rainy", "is_windy", "is_hot", "is_cold"]
#     w = w[[c for c in keep if c in w.columns]].drop_duplicates(subset=["Datum"])
#     return w

# def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
#     """Voegt afgeleide features toe en merge't (optioneel) weerfeatures op Datum."""
#     df = df_in.copy()

#     # Datum als datetime (dag-eerst toestaan voor examenformaat)
#     df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce", dayfirst=True)

#     # Kalenderfeatures
#     df["weekday"]    = df["Datum"].dt.weekday
#     df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
#     df["month"]      = df["Datum"].dt.month

#     # Prime time (20–23)
#     start_fill = df["Start"].fillna(-1)
#     df["prime_time"] = ((start_fill >= 20*60) & (start_fill < 23*60)).astype(int)

#     # Startbin
#     df["start_bin"] = _start_bin_from_minutes(df["Start"])

#     # Categorical netjes
#     df["Programma"] = df["Programma"].astype(str)
#     df["Zender"]    = df["Zender"].astype(str)

#     # Weerdata (optioneel)
#     w = _load_weather(WEATHER_PATH)
#     if w is not None:
#         # merge op Datum (left); als er geen match is blijven NaNs staan (imputer vangt op)
#         df = df.merge(w, how="left", on="Datum")

#     # Zorg dat alle kolommen bestaan
#     out = df.copy()
#     for c in FEATURE_ORDER:
#         if c not in out.columns:
#             out[c] = pd.NA

#     return out[["Datum"] + FEATURE_ORDER]

# if __name__ == "__main__":
#     # Snelle self-check: toont of weather mee gemerged wordt
#     import sys
#     src = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.csv"
#     dfi = pd.read_csv(src, sep=";")
#     f = build_features(dfi)
#     has = {k: (k in f.columns) for k in ["rain_mm","temp_max_c","sunshine_h"]}
#     print("rows:", len(f), "| weather present:", has)
#     print(f[["Datum","rain_mm","temp_max_c","sunshine_h"]].head())


from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# === Basis features ===
CATEGORICAL = ["Programma", "Zender", "start_bin"]
NUMERIC_BASE = ["Start", "Duur", "weekday", "is_weekend", "month", "prime_time"]


NUMERIC = NUMERIC_BASE
FEATURE_ORDER = CATEGORICAL + NUMERIC  # consistente volgorde voor train/predict


def _start_bin_from_minutes(m: pd.Series) -> pd.Categorical:
    bins   = [-1, 6*60, 12*60, 17*60, 20*60, 23*60, 24*60]
    labels = ["nacht", "ochtend", "middag", "vooravond", "prime", "laat"]
    return pd.cut(m, bins=bins, labels=labels)


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Voegt afgeleide features toe en merge't (optioneel) weerfeatures op Datum."""
    df = df_in.copy()

    # Datum als datetime (dag-eerst toestaan voor examenformaat)
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce", dayfirst=True)

    # Kalenderfeatures
    df["weekday"]    = df["Datum"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["month"]      = df["Datum"].dt.month

    # Prime time (20–23)
    start_fill = df["Start"].fillna(-1)
    df["prime_time"] = ((start_fill >= 20*60) & (start_fill < 23*60)).astype(int)

    # Startbin
    df["start_bin"] = _start_bin_from_minutes(df["Start"])

    # Categorical netjes
    df["Programma"] = df["Programma"].astype(str)
    df["Zender"]    = df["Zender"].astype(str)

    # Zorg dat alle kolommen bestaan
    out = df.copy()
    for c in FEATURE_ORDER:
        if c not in out.columns:
            out[c] = pd.NA

    return out[["Datum"] + FEATURE_ORDER]

if __name__ == "__main__":
    # Snelle self-check: toont of weather mee gemerged wordt
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.csv"
    dfi = pd.read_csv(src, sep=";")
    f = build_features(dfi)
    has = {k: (k in f.columns) for k in ["rain_mm","temp_max_c","sunshine_h"]}
    print("rows:", len(f), "| weather present:", has)
    print(f[["Datum","rain_mm","temp_max_c","sunshine_h"]].head())


