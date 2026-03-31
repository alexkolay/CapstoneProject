import pandas as pd

# The IQR model works well for spacetrack_starlink.json because the satellite orbital parameters, 
# such as eccentricity, inclination, and mean motion, follow consistent ranges for satellites 
# within the same constellation. Satellites that differ significantly from these typical 
# ranges may indicate unusual behavior or potential anomalies. The IQR method is effective 
# because it does not assume any specific distribution and is robust to extreme values, making 
# it suitable for real-world satellite data. Additionally, it allows to clearly identify 
# which orbital parameters contributed to each detected anomaly.

# run by staying only in CapstoneProject directory and then running python3 interquartile_range/interquartile_range.py
# used for running with spacetrack_starlink.json

df = pd.read_json("spacetrack_starlink.json")

# use only the main anomaly-driving features
features = [
    "ECCENTRICITY",
    "INCLINATION",
    "BSTAR",
    "MEAN_MOTION"
]

# convert columns to numeric, this ensures calculations work properly
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# store IQR bounds for each feature to explain anomalies later
bounds = {}

# calculate IQR bounds for each feature
for col in features:
    q1 = df[col].quantile(0.25) # first quartile (25th percentile)
    q3 = df[col].quantile(0.75) # third quartile (75th percentile)
    iqr = q3 - q1  # interquartile range = Q3 - Q1

    # define outliers as points outside 3 * IQR from the quartiles (can abjust multiplier for sensitivity)
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    bounds[col] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

# flag anomalies for each feature
# If value is outside the IQR bounds, mark as anomaly (1)
for col in features:
    lower = bounds[col]["lower_bound"]
    upper = bounds[col]["upper_bound"]
    df[f"{col}_is_anomaly"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

# total anomaly score
anomaly_flag_cols = [f"{col}_is_anomaly" for col in features]
df["anomaly_score"] = df[anomaly_flag_cols].sum(axis=1)

# explain which features caused anomaly
def get_anomaly_factors(row):
    factors = []
    for col in features:
        if row[f"{col}_is_anomaly"] == 1:
            lower = bounds[col]["lower_bound"]
            upper = bounds[col]["upper_bound"]
            value = row[col]

            if pd.notna(value):
                if value < lower:
                    factors.append(f"{col}=LOW ({value:.6f} < {lower:.6f})")
                elif value > upper:
                    factors.append(f"{col}=HIGH ({value:.6f} > {upper:.6f})")
    return "; ".join(factors)

df["anomaly_factors"] = df.apply(get_anomaly_factors, axis=1)

# keep only stronger anomalies (can adjust threshold for sensitivity)
anomalies_df = df[df["anomaly_score"] >= 1].copy()

# output columns (can include more if desired, but these are the main orbital parameters, change if features are adjusted)
# keep OBJECT_NAME, OBJECT_ID for identification, anomaly_score and anomaly_factors for explanation, and the main features for context
output_columns = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "anomaly_score",
    "anomaly_factors",
    "ECCENTRICITY",
    "INCLINATION",
    "BSTAR",
    "MEAN_MOTION"
]

# filter to output columns
anomalies_df = anomalies_df[output_columns]

# sort strongest anomalies first
anomalies_df = anomalies_df.sort_values(
    by=["anomaly_score", "OBJECT_NAME"],
    ascending=[False, True]
)

# save output to CSV
anomalies_df.to_csv("interquartile_range/anomalies_interquartile_range.csv", index=False)

print("Done.")
print("Anomalies saved to anomalies_interquartile_range.csv")
print("Number of anomalous satellites:", len(anomalies_df))
print(anomalies_df.head(20).to_string(index=False))