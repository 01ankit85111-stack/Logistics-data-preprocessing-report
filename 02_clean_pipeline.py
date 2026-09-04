"""
02_clean_pipeline.py
---------------------
Data Cleaning & Preprocessing pipeline for the logistics shipment dataset.

Stages:
  1. Load & initial inspection
  2. Structural cleaning (duplicates, text standardization, date parsing)
  3. Missing value treatment
  4. Outlier detection & treatment
  5. Feature engineering (TransitDays)
  6. Normalization / scaling
  7. Save cleaned dataset + summary report
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

# ------------------------------------------------------------------
# 1. LOAD & INITIAL INSPECTION
# ------------------------------------------------------------------
df = pd.read_csv("raw_logistics_data.csv")
print("Raw shape:", df.shape)
print(df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())

report_lines = []
report_lines.append(f"Raw dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
report_lines.append(f"Missing values (raw):\n{df.isna().sum().to_string()}")

# ------------------------------------------------------------------
# 2. STRUCTURAL CLEANING
# ------------------------------------------------------------------

# 2a. Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
report_lines.append(f"Duplicate rows removed: {before - len(df)}")

# 2b. Standardize text columns: strip whitespace, fix casing
text_cols = ["OriginCity", "DestinationCity", "Carrier", "TransportMode",
             "Warehouse", "DeliveryStatus"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()
    df[col] = df[col].replace("Nan", np.nan)

# 2c. Parse OrderDate which has two mixed formats (YYYY-MM-DD and DD/MM/YYYY)
def parse_mixed_date(value):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

df["OrderDate"] = df["OrderDate"].apply(parse_mixed_date)
df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

# 2d. Fix impossible negative distances (data entry sign errors)
neg_mask = df["Distance_km"] < 0
report_lines.append(f"Negative distance values corrected (sign flipped): {neg_mask.sum()}")
df.loc[neg_mask, "Distance_km"] = df.loc[neg_mask, "Distance_km"].abs()

# ------------------------------------------------------------------
# 3. MISSING VALUE TREATMENT
# ------------------------------------------------------------------

# 3a. Categorical: Carrier -- impute with mode (most frequent carrier),
#     since a missing carrier is likely a logging gap, not "no carrier".
carrier_mode = df["Carrier"].mode()[0]
df["Carrier"] = df["Carrier"].fillna(carrier_mode)

# 3b. Numerical: Weight_kg, Distance_km, ShippingCost_INR -- impute with
#     the MEDIAN (robust to skew/outliers) grouped by TransportMode,
#     since these vary systematically by mode (Air vs Road vs Sea).
for col in ["Weight_kg", "Distance_km", "ShippingCost_INR"]:
    df[col] = df.groupby("TransportMode")[col].transform(
        lambda s: s.fillna(s.median())
    )

# 3c. CustomerRating: missing likely means "no feedback given" -- impute
#     with median rating rather than dropping rows (preserves sample size).
df["CustomerRating"] = df["CustomerRating"].fillna(df["CustomerRating"].median())

report_lines.append(f"Missing values after imputation:\n{df.isna().sum().to_string()}")

# ------------------------------------------------------------------
# 4. OUTLIER DETECTION & TREATMENT (IQR method)
# ------------------------------------------------------------------

def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_summary = {}
for col in ["Weight_kg", "Distance_km", "ShippingCost_INR"]:
    low, high = iqr_bounds(df[col])
    n_outliers = ((df[col] < low) | (df[col] > high)).sum()
    outlier_summary[col] = (low, high, n_outliers)
    # Cap (winsorize) rather than delete, to preserve sample size for logistics analysis
    df[col] = df[col].clip(lower=low, upper=high)

for col, (low, high, n) in outlier_summary.items():
    report_lines.append(f"{col}: {n} outliers capped to range [{low:.2f}, {high:.2f}]")

# ------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ------------------------------------------------------------------
df["TransitDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
# Only meaningful for delivered shipments; keep NaN otherwise
df.loc[df["DeliveryStatus"] != "Delivered", "TransitDays"] = np.nan

# ------------------------------------------------------------------
# 6. NORMALIZATION / SCALING
# ------------------------------------------------------------------
# Min-Max normalization -> range [0,1], good for distance-based ML models
for col in ["Weight_kg", "Distance_km", "ShippingCost_INR"]:
    mn, mx = df[col].min(), df[col].max()
    df[col + "_MinMax"] = (df[col] - mn) / (mx - mn)

# Z-score standardization -> mean 0, std 1, good for linear models / PCA
for col in ["Weight_kg", "Distance_km", "ShippingCost_INR"]:
    mu, sigma = df[col].mean(), df[col].std()
    df[col + "_Zscore"] = (df[col] - mu) / sigma

# ------------------------------------------------------------------
# 7. FINAL VALIDATION & SAVE
# ------------------------------------------------------------------
report_lines.append(f"Final cleaned dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
report_lines.append(f"Remaining missing values (post-clean):\n{df.isna().sum().to_string()}")

df.to_csv("cleaned_logistics_data.csv", index=False)

with open("cleaning_summary_report.txt", "w") as f:
    f.write("\n\n".join(report_lines))

print("\n".join(report_lines))
print("\nSaved: cleaned_logistics_data.csv, cleaning_summary_report.txt")
