# LEO Satellite Anomaly Detection

Capstone project exploring anomaly detection for Low Earth Orbit (LEO) satellite constellations (Starlink, OneWeb, and similar). The goal is to learn what "normal" satellite behavior looks like and flag things that stand out, such as:

- RF/telemetry anomalies on the optical/signal side
- Unusual orbital elements (possible spoofed TLE data, drift, maneuvers)
- Outages or link failures

The project works with two fairly different datasets, so there are two separate pipelines.

## Data

- `OPTSAT_dataset.csv` — labeled optical satellite telemetry (OPTSAT-AD), used for supervised anomaly detection.
- `spacetrack_data.csv` / `spacetrack_starlink.json` — orbital element (TLE/GP) data pulled from Space-Track for active satellites. This is the primary orbital dataset used in the pipeline.
- `fetch_constellations.py` — pulls active satellite GP data from the Space-Track API for several constellations (Starlink, OneWeb, Iridium, Planet, etc.). Requires a free Space-Track account.

## Pipelines

**OPTSAT (supervised)**
1. `preprocess_OPTSAT.py` — cleans and encodes `OPTSAT_dataset.csv`, one-hot encodes channels, scales features, and saves everything to `optsat_preprocessed.pkl`.
2. `model_OPTSAT.py` — trains and compares Logistic Regression, Linear SVM, Random Forest, and Gradient Boosting on the preprocessed data. Reports precision/recall/F1 per channel and saves the best model plus test predictions to `optsat_test_predictions.csv`.

**Space-Track (unsupervised)**
1. `preprocess_spacetrack.py` — loads the Space-Track JSON, keeps only active (non-decayed) satellites, derives orbital features (altitude range, mean altitude, etc.), and saves a train/test split to `spacetrack_preprocessed.pkl`.
2. `model_spacetrack.py` — since there are no anomaly labels here, this trains Isolation Forest and One-Class SVM instead and compares which satellites each model flags. Outputs anomaly scores to `spacetrack_anomaly_scores.csv`.

## Running it

Each pipeline needs its preprocessing step run first:

```
python preprocess_OPTSAT.py
python model_OPTSAT.py

python preprocess_spacetrack.py
python model_spacetrack.py
```

Trained models are pickled to `best_optsat_model.pkl` and `best_spacetrack_model.pkl`.

## Notes

- The two pipelines are independent — OPTSAT is supervised (has ground-truth anomaly labels), Space-Track is not, so the modeling approach differs on purpose.
- `fetch_constellations.py` needs Space-Track credentials passed as command-line args; see the docstring at the top of the file for usage.
