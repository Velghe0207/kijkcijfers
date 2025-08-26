import pandas as pd

def parse_number_eu(val):
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().replace("\xa0","").replace(" ","")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def parse_viewers(val):
    f = parse_number_eu(val)
    if f is None: return None
    v = int(round(f))
    return v if 0 <= v <= 5_000_000 else None

df = pd.read_csv("data/processed/train.csv", sep=";")
src = "rateInK" if "rateInK" in df.columns else ("rateInKAll" if "rateInKAll" in df.columns else None)
if src:
    df["Kijkcijfers"] = df[src].apply(parse_viewers)

# houd enkel daily + north
if "period" in df.columns:
    df = df[df["period"].astype(str).str.lower() == "daily"]
if "reportType" in df.columns:
    df = df[df["reportType"].astype(str).str.lower() == "north"]

# veiligheidsnet tegen outliers
df = df[df["Kijkcijfers"].notna() & df["Kijkcijfers"].between(0, 5_000_000)]

df.to_csv("data/processed/train.csv", sep=";", index=False)
print("Hersteld → data/processed/train.csv")
