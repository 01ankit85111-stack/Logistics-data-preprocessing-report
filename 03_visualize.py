import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw = pd.read_csv("raw_logistics_data.csv")
clean = pd.read_csv("cleaned_logistics_data.csv")

plt.rcParams.update({"font.size": 10})

# 1. Missing values before vs after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
raw.isna().sum().sort_values(ascending=False).head(6).plot(kind="bar", ax=axes[0], color="#c0392b")
axes[0].set_title("Missing Values - Raw Data")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)

core_cols = ["Carrier", "Weight_kg", "Distance_km", "ShippingCost_INR", "CustomerRating"]
clean[core_cols].isna().sum().plot(kind="bar", ax=axes[1], color="#27ae60")
axes[1].set_title("Missing Values - Cleaned Data")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("fig_missing_values.png", dpi=150)
plt.close()

# 2. Outlier boxplots before vs after (ShippingCost_INR)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].boxplot(raw["ShippingCost_INR"].dropna(), vert=True)
axes[0].set_title("Shipping Cost - Before Cleaning")
axes[0].set_ylabel("INR")
axes[1].boxplot(clean["ShippingCost_INR"].dropna(), vert=True)
axes[1].set_title("Shipping Cost - After Outlier Capping")
axes[1].set_ylabel("INR")
plt.tight_layout()
plt.savefig("fig_outliers_cost.png", dpi=150)
plt.close()

# 3. Normalized distributions
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(clean["Weight_kg_MinMax"].dropna(), bins=30, alpha=0.6, label="Weight (Min-Max)")
ax.hist(clean["Distance_km_MinMax"].dropna(), bins=30, alpha=0.6, label="Distance (Min-Max)")
ax.hist(clean["ShippingCost_INR_MinMax"].dropna(), bins=30, alpha=0.6, label="Cost (Min-Max)")
ax.set_title("Normalized Feature Distributions (0-1 scale)")
ax.set_xlabel("Normalized value")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("fig_normalized_dist.png", dpi=150)
plt.close()

# 4. Delivery status breakdown
fig, ax = plt.subplots(figsize=(6, 4))
clean["DeliveryStatus"].value_counts().plot(kind="bar", ax=ax, color="#2980b9")
ax.set_title("Shipment Count by Delivery Status (Cleaned Data)")
ax.set_ylabel("Number of Shipments")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("fig_delivery_status.png", dpi=150)
plt.close()

print("Saved 4 figures.")
