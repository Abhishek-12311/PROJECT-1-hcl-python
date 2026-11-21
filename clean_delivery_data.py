"""
Delivery ETA dataset - data cleaning with NumPy + pandas
Input : Delivery_ETA_Dirty_Data.xlsx
Output: Delivery_ETA_Clean_Data.xlsx
"""
import re
import numpy as np
import pandas as pd

INPUT_FILE = "Delivery_ETA_Dirty_Data.xlsx"
OUTPUT_FILE = "Delivery_ETA_Clean_Data.xlsx"

# ------------------------------------------------------------------ load
df = pd.read_excel(INPUT_FILE, dtype=object)
print("Raw shape:", df.shape)
log = {"raw_rows": len(df)}

# ------------------------------------------------------------------ 1. strip spaces, unify missing markers
MISSING = {"", "n/a", "na", "null", "none", "-", "nan"}

def to_nan(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip()
    return np.nan if s.lower() in MISSING else s

df = df.map(to_nan)               # use applymap() on pandas < 2.1

# ------------------------------------------------------------------ 2. remove duplicates
df = df.drop_duplicates()                                   # exact copies
df = df.drop_duplicates(subset="Delivery_ID", keep="first") # same ID
log["duplicates_removed"] = log["raw_rows"] - len(df)

# ------------------------------------------------------------------ 3. text columns
df["Partner_Name"] = df["Partner_Name"].str.title().fillna("Unknown")

city_map = {"ghaziabad": "Ghaziabad", "ghazibad": "Ghaziabad", "gzb": "Ghaziabad",
            "noida": "Noida",
            "delhi": "Delhi", "new delhi": "Delhi",
            "gurgaon": "Gurgaon", "gurugram": "Gurgaon",
            "faridabad": "Faridabad", "faridabd": "Faridabad"}
df["City"] = df["City"].str.lower().map(city_map).fillna("Unknown")

traffic_map = {"low": "Low", "lo": "Low",
               "medium": "Medium", "med": "Medium",
               "high": "High", "hi": "High"}
df["Traffic_Level"] = df["Traffic_Level"].str.lower().map(traffic_map)

vehicle_map = {"bike": "Bike", "2-wheeler": "Bike", "scooter": "Scooter", "van": "Van"}
df["Vehicle_Type"] = df["Vehicle_Type"].str.lower().map(vehicle_map).fillna("Unknown")

weather_map = {"clear": "Clear", "rain": "Rain", "rainy": "Rain", "fog": "Fog"}
df["Weather"] = df["Weather"].str.lower().map(weather_map).fillna("Unknown")

# ------------------------------------------------------------------ 4. dates (mixed formats)
def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).split(" 00:00")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT

df["Order_Date"] = df["Order_Date"].map(parse_date)

# ------------------------------------------------------------------ 5. numeric columns
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

def to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    if s in WORDS:
        return float(WORDS[s])
    s = s.replace(",", ".")                 # 7,7 -> 7.7
    m = re.search(r"-?\d+(\.\d+)?", s)      # drop units: km, min, pcs
    return float(m.group()) if m else np.nan

for col in ["Distance_km", "Package_Count", "Delivery_Time_min"]:
    df[col] = df[col].map(to_number)

# ------------------------------------------------------------------ 6. invalid values / outliers -> NaN
df["Distance_km"] = df["Distance_km"].where(df["Distance_km"].between(0.1, 100))
df["Package_Count"] = df["Package_Count"].where(df["Package_Count"].between(1, 50))
df["Delivery_Time_min"] = df["Delivery_Time_min"].where(df["Delivery_Time_min"].between(1, 300))

# IQR check on the target for anything extreme that is still left
q1, q3 = df["Delivery_Time_min"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 3 * iqr
df.loc[df["Delivery_Time_min"] > upper, "Delivery_Time_min"] = np.nan

# ------------------------------------------------------------------ 7. missing values
# target missing -> drop (we should not invent the label we want to predict)
before = len(df)
df = df.dropna(subset=["Delivery_Time_min"])
log["rows_dropped_no_target"] = before - len(df)

log["missing_distance_filled"] = int(df["Distance_km"].isna().sum())
log["missing_packages_filled"] = int(df["Package_Count"].isna().sum())
log["missing_traffic_filled"] = int(df["Traffic_Level"].isna().sum())

df["Distance_km"] = df["Distance_km"].fillna(df.groupby("City")["Distance_km"].transform("median"))
df["Distance_km"] = df["Distance_km"].fillna(df["Distance_km"].median())
df["Package_Count"] = df["Package_Count"].fillna(df["Package_Count"].median())
df["Traffic_Level"] = df["Traffic_Level"].fillna(df["Traffic_Level"].mode()[0])

# ------------------------------------------------------------------ 8. final types, order
df["Distance_km"] = df["Distance_km"].round(1)
df["Package_Count"] = df["Package_Count"].round().astype(int)
df["Delivery_Time_min"] = df["Delivery_Time_min"].round().astype(int)
df["Traffic_Level"] = pd.Categorical(df["Traffic_Level"], ["Low", "Medium", "High"], ordered=True)

df = df.sort_values(["Order_Date", "Delivery_ID"]).reset_index(drop=True)
df["Order_Date"] = df["Order_Date"].dt.date
log["clean_rows"] = len(df)

# ------------------------------------------------------------------ save + quick report
df.to_excel(OUTPUT_FILE, index=False)
print("Clean shape:", df.shape)
print(log)
print(df.describe(include="all").T)
