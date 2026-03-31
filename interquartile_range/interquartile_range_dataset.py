import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# run by staying only in CapstoneProject directory and then running python3 interquartile_range/interquartile_range_dataset.py
# used for testing with dataset.csv
# accuracy is printed at the top of the output, NOT in the .csv output
# got accuracy of: 0.7753179463024022

df = pd.read_csv("dataset.csv")

features = [
    "mean",
    "std",
    "var",
    "skew",
    "kurtosis",
    "diff_var",
    "diff2_var",
    "n_peaks"
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

train_df = df[(df["train"] == 1) & (df["anomaly"] == 0)].copy()
test_df = df.copy()

bounds = {}

for col in features:
    q1 = train_df[col].quantile(0.25)
    q3 = train_df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    bounds[col] = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

for col in features:
    lower = bounds[col]["lower_bound"]
    upper = bounds[col]["upper_bound"]
    test_df[f"{col}_is_anomaly"] = ((test_df[col] < lower) | (test_df[col] > upper)).astype(int)

anomaly_flag_cols = [f"{col}_is_anomaly" for col in features]
test_df["anomaly_score"] = test_df[anomaly_flag_cols].sum(axis=1)

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

test_df["anomaly_factors"] = test_df.apply(get_anomaly_factors, axis=1)

test_df["predicted_anomaly"] = (test_df["anomaly_score"] >= 1).astype(int)

# evaluation
y_true = test_df["anomaly"]
y_pred = test_df["predicted_anomaly"]

print("Accuracy:", accuracy_score(y_true, y_pred))

anomalies_df = test_df[test_df["predicted_anomaly"] == 1].copy()

output_columns = [
    "segment",
    "anomaly",
    "predicted_anomaly",
    "anomaly_score",
    "anomaly_factors"
] + features

anomalies_df = anomalies_df[output_columns]

anomalies_df = anomalies_df.sort_values(
    by=["anomaly_score", "segment"],
    ascending=[False, True]
)

anomalies_df.to_csv("interquartile_range/anomalies_interquartile_range_dataset.csv", index=False)

print("\nDone.")
print("Anomalies saved to anomalies_interquartile_range_dataset.csv")
print("Number of predicted anomalies:", len(anomalies_df))
print(anomalies_df.head(20).to_string(index=False))