import pandas as pd
import numpy as np
from scipy.stats import zscore

# 10-day retail sales data (Rs. '000), 2 missing, 1 spike
df = pd.DataFrame({
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
})

# (a) Missing values and median fill
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
df["Revenue"] = df["Revenue"].fillna(df["Revenue"].median())
print("After median fill:")
print(df)

# (b) Min-Max normalization and Z-score standardization
rev = df["Revenue"]
df["Revenue_MinMax"] = (rev - rev.min()) / (rev.max() - rev.min())
df["Revenue_Zscore"] = zscore(rev)

# (c) Outlier detection using the IQR method
Q1 = rev.quantile(0.25)
Q3 = rev.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["Outlier"] = (rev < lower) | (rev > upper)

print("IQR bounds:", lower, "-", upper)
print("IQR outliers detected on day(s):", df[df["Outlier"]]["Day"].tolist())

# (d) Final table
print(df)