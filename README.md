We will build a machine learning (ML) system that learns “normal” behavior of a Low Earth Orbit (LEO) satellite constellation (e.g., Starlink, NOAA, or SatNOGS telemetry), then automatically detects anomalies that may include:
- Radio frequency (RF) interference or jamming
- Spoofed TLE data / position anomalies
- Satellite outages / Satellite-to-Ground link failures
- Multi-party routing behavior


Notes specific to this branch:
There are two additions to this branch from the main branch.
- One is the rf_and_xgb.py file, which is, as you'd expect, a python file that runs both a random forest model and an XGBoost model.
- The other change is just a renaming of the file dataset.csv, which is now called OPTSAT_data.csv. 

rf_and_xgb.py runs the models based on the OPTSAT_data.csv, and creates 5 files. The primary files are data_summary.json, model_results.json, and report.txt.

Output Files:
- data-summary.json: a descrition of the data, including basic information like the number of rows, columns, names of unique channels (satellites), and missing values per column.
- model_results.json: the basic evaluation metrics for both the random forest model and the XGBoost model. It also includes the "cv_scores", which are just the same scores after cross validation.
   - I separated the regular metrics from the metrics after cross validation because the training data was pre-labeled in the dataset, and I wasn't sure how important it was that that data remains the training data. Can't really think of a reason why we'd have to only use that data for training except that maybe some features were calculated based on groups and this is to prevent data leakage...? idk. 
