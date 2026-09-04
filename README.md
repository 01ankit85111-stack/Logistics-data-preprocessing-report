# Logistics Data Collection & Preprocessing Pipeline

Internship task: a complete data preprocessing pipeline for a logistics shipment
dataset — data cleaning, missing-value handling, outlier treatment, and
normalization — built with Python (pandas, NumPy, matplotlib).

## 📁 Project Structure

```
.
├── 01_generate_dataset.py            # Simulates raw shipment data collection
├── 02_clean_pipeline.py              # Full cleaning & preprocessing pipeline
├── 03_visualize.py                   # Generates report figures
├── raw_logistics_data.csv            # Raw (uncleaned) dataset — 1,225 rows x 13 cols
├── cleaned_logistics_data.csv        # Cleaned dataset — 1,202 rows x 20 cols
├── cleaning_summary_report.txt       # Log of every transformation applied
├── Logistics_Data_Preprocessing_Report.docx   # Full written report
└── README.md                         # This file
```

## 📊 Dataset

Shipment-level logistics data with the following columns:

| Column | Description |
|---|---|
| `OrderID` | Unique shipment identifier |
| `OrderDate` / `DeliveryDate` | Order placement / delivery dates |
| `OriginCity` / `DestinationCity` | Shipment route |
| `Carrier` | Logistics carrier (BlueDart, Delhivery, DHL, FedEx, Ekart, XpressBees) |
| `TransportMode` | Road / Rail / Air / Sea |
| `Warehouse` | Dispatching warehouse |
| `Weight_kg` | Shipment weight |
| `Distance_km` | Distance travelled |
| `ShippingCost_INR` | Shipping cost |
| `CustomerRating` | Customer satisfaction rating (1–5) |
| `DeliveryStatus` | Delivered / Delayed / In Transit / Cancelled |

The raw dataset was generated to mirror real-world logistics exports (e.g.
Kaggle/DataCo-style supply chain datasets) and intentionally includes common
data quality problems: missing values, duplicate rows, inconsistent text
casing/whitespace, mixed date formats, sign errors, and outliers.

## ⚙️ How to Run

Requires Python 3 with `pandas`, `numpy`, and `matplotlib` installed:

```bash
pip install pandas numpy matplotlib

python 01_generate_dataset.py   # -> raw_logistics_data.csv
python 02_clean_pipeline.py     # -> cleaned_logistics_data.csv + cleaning_summary_report.txt
python 03_visualize.py          # -> fig_*.png (used in the report)
```

## 🧹 Pipeline Stages

1. **Structural cleaning** — remove duplicates, standardize text casing/whitespace,
   parse mixed date formats, fix sign errors.
2. **Missing value treatment** — mode imputation for categorical fields,
   mode-grouped median imputation for numeric fields, median imputation for
   ratings; structural nulls (e.g. `DeliveryDate` for undelivered shipments)
   are preserved rather than imputed.
3. **Outlier detection & treatment** — IQR method (`Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`),
   values winsorized (capped) rather than deleted to preserve sample size.
4. **Feature engineering** — derived `TransitDays` (delivery date − order date).
5. **Normalization** — Min-Max scaling (`_MinMax` columns, range [0,1]) and
   Z-score standardization (`_Zscore` columns, mean 0 / std 1).

## 📈 Results

| Metric | Before | After |
|---|---|---|
| Rows | 1,225 | 1,202 |
| Duplicate rows | 23 | 0 |
| Missing values (all numeric/categorical cols) | 244 total | 0 |
| Negative distance values | 5 | 0 |
| Outliers (Weight_kg / Cost, IQR) | 59 / 38 | 0 (capped) |
| Columns | 13 | 20 |

See `Logistics_Data_Preprocessing_Report.docx` for the full write-up, including
methodology, reasoning behind each technique, charts, and a reflection on how
data quality affects logistics decision-making.

## 🛠️ Tools Used

Python 3 · pandas · NumPy · matplotlib

## ✍️ Author

Ankit — B.Tech CSE (AI & ML), NIAT, Prayagraj (Batch 2025–2029)
