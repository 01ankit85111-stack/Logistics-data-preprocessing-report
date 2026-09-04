"""
01_generate_dataset.py
-----------------------
Simulates the "Data Collection" stage of the pipeline.

In a real internship setting you would collect this data from a source such as:
  - Kaggle "Logistics and Supply Chain Dataset" / "DataCo Smart Supply Chain Dataset"
  - A company's Warehouse Management System (WMS) or Transport Management System (TMS) export
  - Public shipment tracking APIs (e.g., carrier tracking exports)

Since we cannot download a live dataset in this environment, this script SIMULATES a
realistic raw logistics export -- a shipment-level dataset with the same structure,
value ranges, and (importantly) the same kinds of data quality problems that a real
logistics dataset would contain: missing values, duplicate rows, inconsistent text
formatting, outliers, and mixed units.

Output: raw_logistics_data.csv  (this is our "collected" dataset)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 1200  # number of shipment records

cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
          "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
carriers = ["BlueDart", "Delhivery", "DHL", "FedEx", "Ekart", "XpressBees"]
modes = ["Road", "Rail", "Air", "Sea"]
warehouses = ["WH-North", "WH-South", "WH-East", "WH-West", "WH-Central"]

start_date = datetime(2024, 1, 1)

rows = []
for i in range(N):
    order_id = f"ORD{10000 + i}"
    origin = np.random.choice(cities)
    destination = np.random.choice([c for c in cities if c != origin])
    carrier = np.random.choice(carriers)
    mode = np.random.choice(modes, p=[0.55, 0.15, 0.20, 0.10])

    order_date = start_date + timedelta(days=int(np.random.uniform(0, 365)))
    # Base transit time depends on mode
    base_transit = {"Road": 4, "Rail": 6, "Air": 1.5, "Sea": 12}[mode]
    transit_days = max(0.5, np.random.normal(base_transit, base_transit * 0.3))
    delivery_date = order_date + timedelta(days=transit_days)

    # Weight in kg -- occasionally recorded in grams by mistake (data entry error)
    weight_kg = np.round(np.random.gamma(shape=2.0, scale=15.0), 2)

    # Distance in km, correlated loosely with transit time
    distance_km = np.round(transit_days * np.random.uniform(60, 140), 1)

    # Shipping cost roughly a function of weight & distance, with noise
    cost = round(50 + 0.08 * distance_km + 4.5 * weight_kg + np.random.normal(0, 40), 2)

    # Customer rating 1-5
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.17, 0.35, 0.35])

    warehouse = np.random.choice(warehouses)

    status = np.random.choice(
        ["Delivered", "Delivered", "Delivered", "Delayed", "In Transit", "Cancelled"],
        p=[0.55, 0.15, 0.1, 0.1, 0.06, 0.04]
    )

    rows.append({
        "OrderID": order_id,
        "OrderDate": order_date.strftime("%Y-%m-%d"),
        "DeliveryDate": delivery_date.strftime("%Y-%m-%d") if status == "Delivered" else "",
        "OriginCity": origin,
        "DestinationCity": destination,
        "Carrier": carrier,
        "TransportMode": mode,
        "Warehouse": warehouse,
        "Weight_kg": weight_kg,
        "Distance_km": distance_km,
        "ShippingCost_INR": cost,
        "CustomerRating": rating,
        "DeliveryStatus": status,
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------
# Now we deliberately INJECT realistic data quality problems, the
# same kinds of problems a real-world logistics export would have.
# ---------------------------------------------------------------
rng = np.random.default_rng(7)

# 1) Missing values in several columns (MCAR / MAR simulation)
for col, frac in [("Weight_kg", 0.04), ("Distance_km", 0.03),
                   ("ShippingCost_INR", 0.05), ("CustomerRating", 0.06),
                   ("Carrier", 0.02)]:
    idx = rng.choice(df.index, size=int(frac * len(df)), replace=False)
    df.loc[idx, col] = np.nan

# 2) Inconsistent text casing / whitespace (common export issue)
messy_idx = rng.choice(df.index, size=80, replace=False)
df.loc[messy_idx, "OriginCity"] = df.loc[messy_idx, "OriginCity"].str.lower()
messy_idx2 = rng.choice(df.index, size=60, replace=False)
df.loc[messy_idx2, "Carrier"] = df.loc[messy_idx2, "Carrier"].apply(lambda x: f"  {x}  " if pd.notna(x) else x)

# 3) Duplicate rows (system re-export / double entry)
dupes = df.sample(25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 4) Outliers: a few absurd weight and cost values (sensor/typo errors)
outlier_idx = rng.choice(df.index, size=10, replace=False)
df.loc[outlier_idx, "Weight_kg"] = df.loc[outlier_idx, "Weight_kg"] * rng.uniform(15, 30)
outlier_idx2 = rng.choice(df.index, size=8, replace=False)
df.loc[outlier_idx2, "ShippingCost_INR"] = df.loc[outlier_idx2, "ShippingCost_INR"] * rng.uniform(10, 20)

# 5) A few negative / impossible values (data entry errors)
neg_idx = rng.choice(df.index, size=5, replace=False)
df.loc[neg_idx, "Distance_km"] = -df.loc[neg_idx, "Distance_km"]

# 6) Inconsistent date formats for a subset of OrderDate (system migration artifact)
alt_fmt_idx = rng.choice(df.index, size=30, replace=False)
def to_alt_format(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        return d
df.loc[alt_fmt_idx, "OrderDate"] = df.loc[alt_fmt_idx, "OrderDate"].apply(to_alt_format)

# Shuffle rows to mimic an unordered raw export
df = df.sample(frac=1.0, random_state=99).reset_index(drop=True)

df.to_csv("raw_logistics_data.csv", index=False)
print(f"Generated raw_logistics_data.csv with {len(df)} rows and {df.shape[1]} columns")
print(df.isna().sum())
