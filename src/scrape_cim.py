from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.cim.be/api/tv_public_results"
RAW_DIR = Path("data/raw/cim/pages")
INTERIM_PATH = Path("data/interim/cim_daily.parquet")
PROCESSED_PATH = Path("data/processed/train.csv")
USER_AGENT = "kijkcijfers-student/1.0 (+hogent project)"

PROCESSED_COL_ORDER = ["Programma","Zender","Datum","Start","Duur","Kijkcijfers","ranking","period","reportType"]

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

def _to_minutes(s: Optional[str]) -> Optional[int]:
    if not s or not isinstance(s, str): return None
    s = s.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        # numeric-as-string?
        try: return int(float(s))
        except Exception: return None
    h, mi = int(m.group(1)), int(m.group(2))
    sec = int(m.group(3)) if m.group(3) else 0
    return h*60 + mi + (1 if sec >= 30 else 0)

def _clean_int(val) -> Optional[int]:
    if val is None: return None
    digits = re.sub(r"\D", "", str(val))
    return int(digits) if digits else None

def _save_raw(page: int, payload) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, list):
        payload = {"hydra:member": payload}
    (RAW_DIR / f"page_{page}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def _load_raw(page: int) -> Optional[Dict[str, Any]]:
    p = RAW_DIR / f"page_{page}.json"
    if not p.exists(): return None
    try:
        payload = json.loads(p.read_text())
        if isinstance(payload, list):
            payload = {"hydra:member": payload}
        return payload
    except Exception:
        return None

def fetch_page(page: int, session: Optional[requests.Session] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
    if use_cache:
        cached = _load_raw(page)
        if cached is not None:
            return cached.get("hydra:member", [])
    sess = session or _make_session()
    url = f"{BASE_URL}?reportType=north&page={page}"
    r = sess.get(url, timeout=20)
    if r.status_code == 200:
        payload = r.json()
        if isinstance(payload, list):
            members = payload
        elif isinstance(payload, dict):
            members = payload.get("hydra:member") or payload.get("results") or payload.get("data") or []
        else:
            members = []
        _save_raw(page, {"hydra:member": members})
        return members
    return []

def _normalize(members: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in members:
        programma = (e.get("description") or e.get("description2") or e.get("title") or "").strip()
        zender    = (e.get("channel") or e.get("station") or e.get("broadcaster") or "").strip()
        kijkers   = _clean_int(e.get("rateInK")) or _clean_int(e.get("rateInKAll")) or _clean_int(e.get("live"))
        datum_str = e.get("dateResult") or e.get("dateDiff") or ""
        start     = _to_minutes(e.get("startTime"))
        duur      = _to_minutes(e.get("rLength"))
        rows.append({
            "Programma": programma or None,
            "Zender": zender or None,
            "Kijkcijfers": kijkers,
            "Datum": pd.to_datetime(datum_str[:10], errors="coerce"),
            "Start": start, "Duur": duur,
            "ranking": e.get("ranking"),
            "period": e.get("period"),
            "reportType": e.get("reportType"),
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Programma","Zender","Kijkcijfers","Datum"])
    df["Kijkcijfers"] = df["Kijkcijfers"].astype("Int64")
    df = df.drop_duplicates(subset=["Datum","Zender","Programma"], keep="first")
    return df

def collect_pages(start_page: int, end_page: int, stop_when_empty: bool = True) -> pd.DataFrame:
    _ensure_dirs()
    sess = _make_session()
    frames, empty_streak = [], 0
    for page in range(start_page, end_page + 1):
        print(f"[CIM] Pagina {page}…")
        members = fetch_page(page, sess, use_cache=True)
        if not members:
            print("  • leeg")
            empty_streak += 1
            if stop_when_empty and empty_streak >= 3:
                print("  • vroegtijdig stoppen (3× leeg).")
                break
            continue
        empty_streak = 0
        df = _normalize(members)
        if not df.empty: frames.append(df)
        time.sleep(0.15)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PROCESSED_COL_ORDER)
    # merge met bestaande
    if INTERIM_PATH.exists() and not all_df.empty:
        try:
            prev = pd.read_parquet(INTERIM_PATH)
            all_df = pd.concat([prev, all_df], ignore_index=True)
        except Exception:
            pass
    all_df = all_df.drop_duplicates(subset=["Datum","Zender","Programma"]).sort_values(["Datum","Zender","Programma"])
    all_df.to_parquet(INTERIM_PATH, index=False)
    print(f"[OK] Interim → {INTERIM_PATH} ({len(all_df)} rijen)")
    return all_df

def export_processed(interim_path: Path = INTERIM_PATH, out_path: Path = PROCESSED_PATH) -> Path:
    _ensure_dirs()
    df = pd.read_parquet(interim_path) if interim_path.exists() else pd.DataFrame()
    for c in PROCESSED_COL_ORDER:
        if c not in df.columns: df[c] = pd.NA
    df["Datum"] = pd.to_datetime(df["Datum"])
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["Duur"]  = pd.to_numeric(df["Duur"], errors="coerce")
    df["Kijkcijfers"] = pd.to_numeric(df["Kijkcijfers"], errors="coerce").astype("Int64")
    df[PROCESSED_COL_ORDER].to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"[OK] Processed CSV → {out_path} ({len(df)} rijen)")
    return out_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start_page", type=int, required=True)
    p.add_argument("--end_page", type=int, required=True)
    p.add_argument("--no_early_stop", action="store_true")
    args = p.parse_args()
    df = collect_pages(args.start_page, args.end_page, stop_when_empty=not args.no_early_stop)
    if not df.empty:
        export_processed()
