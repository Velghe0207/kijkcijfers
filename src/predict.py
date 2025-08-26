from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.clean import clean_dataframe
from src.features import build_features, FEATURE_ORDER

MODEL_PATH = Path("models/model.joblib")


def main():
    ap = argparse.ArgumentParser(description="Predict kijkcijfers op examen-CSV.")
    ap.add_argument("--input", required=True, help="Pad naar input CSV (Programma;Zender;Datum;Start;Duur)")
    ap.add_argument("--output", required=True, help="Pad naar output CSV met VoorspeldeKijkcijfers")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model niet gevonden: {MODEL_PATH}. Train eerst je model.")

    # 1) Load model bundle
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipe"]
    cal  = bundle.get("cal", {"global": 1.0})

    # 2) Read + clean (zonder target; north-only kan blijven, schaadt niet)
    df_in = pd.read_csv(args.input, sep=";")
    df_clean, rep = clean_dataframe(df_in, require_target=False, keep_only_region="north")

    # 3) Features bouwen in dezelfde volgorde als training
    feat_df = build_features(df_clean)
    X = feat_df[FEATURE_ORDER]

    # 4) Ruwe voorspelling
    preds = pipe.predict(X)

    # 5) Kalibratie toepassen (per zender, anders global)
    z = df_clean["Zender"].astype(str).values
    factors = np.array([ cal.get(zz, cal.get("global", 1.0)) for zz in z ], dtype=float)
    preds_cal = preds * factors

    # 6) Afronden en ondergrens 0
    preds_final = np.maximum(0, np.round(preds_cal / 10.0) * 10).astype(int)

    # 7) Output wegschrijven
    out = df_clean.copy()
    out["VoorspeldeKijkcijfers"] = preds_final
    out.to_csv(args.output, sep=";", index=False, encoding="utf-8-sig")
    print(f"Gereed → {args.output} | n={len(out)}")

if __name__ == "__main__":
    main()
