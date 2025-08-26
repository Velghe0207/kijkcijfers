from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
import joblib
import sklearn

from src.clean import clean_dataframe
from src.features import build_features, FEATURE_ORDER, CATEGORICAL, NUMERIC

DEFAULT_INPUT = Path("data/processed/train2.csv")
MODEL_PATH    = Path("models/model.joblib")  # let op: bundel (pipe + cal + meta)
METRICS_PATH  = Path("reports/metrics.json")


def _build_preprocessor() -> ColumnTransformer:
    """
    OneHotEncoder moet DENSE teruggeven omdat HGB geen sparse slikt.
    Compatibel met sklearn <1.2 (sparse=False) en >=1.2 (sparse_output=False).
    """
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  ohe),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL),
            ("num", num_pipe, NUMERIC),
        ],
        sparse_threshold=0.0  # forceer dense
    )
    return pre


def _compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zenders: np.ndarray,
    min_count: int = 20,
    clip_global: tuple[float, float] = (0.5, 1.5),
    clip_per_channel: tuple[float, float] = (0.4, 1.6),
) -> dict:
    """
    Bepaal een globale kalibratiefactor en (optioneel) per zender, met clipping.
    """
    cal: dict[str, float] = {}
    pred_sum = float(np.sum(y_pred))
    true_sum = float(np.sum(y_true))
    if pred_sum <= 1e-9:
        cal["global"] = 1.0
    else:
        g = true_sum / pred_sum
        cal["global"] = float(np.clip(g, *clip_global))

    # per channel (alleen als genoeg voorbeelden)
    unique = pd.Series(zenders).astype(str).unique()
    for ch in unique:
        m = (zenders == ch)
        if int(np.sum(m)) >= min_count:
            p = float(np.sum(y_pred[m]))
            t = float(np.sum(y_true[m]))
            if p > 1e-9:
                f = t / p
                cal[ch] = float(np.clip(f, *clip_per_channel))
    return cal


def _expanded_feature_names(pre: ColumnTransformer) -> list[str]:
    """
    Haal geëxpandeerde featurenamen op uit de ColumnTransformer:
    - OneHotEncoder kolommen
    - Numerieke kolommen (passthrough)
    """
    # cat
    cat_pipe = pre.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
    # num
    num_names = list(NUMERIC)
    return cat_names + num_names


def train(input_csv: Path = DEFAULT_INPUT) -> dict:
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # 1) Load & clean (north-only, met target)
    raw = pd.read_csv(input_csv, sep=";")
    df, rep = clean_dataframe(raw, require_target=True,
                            keep_only_region="north",
                            keep_only_period="daily")
    print(f"[clean] before={rep.n_rows_before} after={rep.n_rows_after} "
        f"removed_non_daily={rep.n_removed_non_daily} south={rep.n_removed_south} "
        f"missing_target={rep.n_removed_missing_target}")

    if len(df) < 100:
        raise ValueError(f"Te weinig rijen na cleaning ({len(df)}). Scrape wat meer data of versoepel filters.")

    # 2) Sorteer MET target samen en reset index
    df_sorted = df.sort_values("Datum").reset_index(drop=True)

    # 3) Feature build OP de gesorteerde df
    fdf = build_features(df_sorted)
    X_all = fdf[FEATURE_ORDER]
    y_all = df_sorted["Kijkcijfers"].astype(float).to_numpy()
    print("Features:", FEATURE_ORDER)
    print("Feature sample:")
    print(X_all.head(3))

    # 4) Tijdssplit (80/20) op al-gesorteerde data
    split_idx = int(len(df_sorted) * 0.8)
    if split_idx <= 0 or split_idx >= len(df_sorted):
        raise ValueError("Split index ongeldig; dataset te klein of geen variatie in Datum.")
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    # 5) Pipeline + model
    pre = _build_preprocessor()
    model = HistGradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    # 6) Fit
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # 7) Metrics (voor kalibratie)
    mape = float(mean_absolute_percentage_error(y_test, pred))
    mae  = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2   = float(r2_score(y_test, pred))

    # 8) Kalibratie (globaal + per zender met voldoende support)
    z_test = df_sorted["Zender"].iloc[split_idx:].astype(str).to_numpy()
    cal = _compute_calibration(y_test, pred, z_test, min_count=20)

    # 9) Permutation importance (op het model met getransformeerde testfeatures)
    Xt_test = pipe.named_steps["pre"].transform(X_test)
    # score = neg-MAPE (hogere absolute waarde => belangrijker)
    pim = permutation_importance(
        pipe.named_steps["model"],
        Xt_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=None
    )
    feat_names = _expanded_feature_names(pipe.named_steps["pre"])
    # Sorteer aflopend op mean importance
    order = np.argsort(-pim.importances_mean)
    top_k = min(30, len(order))
    perm_top = [
        {"feature": feat_names[i], "importance": float(pim.importances_mean[i])}
        for i in order[:top_k]
    ]

    # 10) Save bundle (pipe + cal + meta)
    bundle = {
        "pipe": pipe,
        "cal": cal,
        "meta": {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "feature_order": FEATURE_ORDER,
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
            "n_rows_clean_before": rep.n_rows_before,
            "n_rows_clean_after": rep.n_rows_after,
            "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        },
    }
    joblib.dump(bundle, MODEL_PATH)

    # 11) Save metrics (incl. perm importance top)
    metrics = {
        "MAPE_raw": mape,
        "MAE_raw":  mae,
        "RMSE_raw": rmse,
        "R2_raw":   r2,
        "calibration": cal,
        "perm_importance_top": perm_top,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Saved model bundle →", MODEL_PATH)
    print("Saved metrics      →", METRICS_PATH)
    print("Eval (raw):", {k: metrics[k] for k in ["MAPE_raw", "MAE_raw", "RMSE_raw", "R2_raw", "n_train", "n_test"]})
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    args = p.parse_args()
    train(Path(args.input))
