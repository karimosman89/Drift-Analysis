import datetime
import pandas as pd
import requests
import zipfile
import io
import warnings

from sklearn import ensemble, model_selection
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def fetch_data() -> pd.DataFrame:
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)), axis=1)
    return raw_data


raw_data = process_data(fetch_data())


target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']


reference_jan11 = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current_feb11 = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    reference_jan11[numerical_features + categorical_features],
    reference_jan11[target],
    test_size=0.3,
    random_state=42
)

# Model training
regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
regressor.fit(X_train, y_train)

# Predictions
preds_train = regressor.predict(X_train)
preds_test = regressor.predict(X_test)

# Add actual target and prediction columns to the training data for later performance analysis
X_train['target'] = y_train
X_train['prediction'] = preds_train

# Add actual target and prediction columns to the test data for later performance analysis
X_test['target'] = y_test
X_test['prediction'] = preds_test

# Column mapping for Evidently
column_mapping = ColumnMapping()
column_mapping.target = 'target'
column_mapping.prediction = 'prediction'
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

# Model validation report
regression_performance_report = Report(metrics=[RegressionPreset()])
regression_performance_report.run(reference_data=X_train.sort_index(), current_data=X_test.sort_index(), column_mapping=column_mapping)
regression_performance_report.save_html('model_validation_report.html')

# Train the production model on the entire January data
regressor.fit(reference_jan11[numerical_features + categorical_features], reference_jan11[target])

# Predictions for the reference data
reference_jan11['target'] = reference_jan11[target]
reference_jan11['prediction'] = regressor.predict(reference_jan11[numerical_features + categorical_features])

# Production model drift report
regression_performance_report = Report(metrics=[RegressionPreset()])
regression_performance_report.run(reference_data=reference_jan11, current_data=reference_jan11, column_mapping=column_mapping)
regression_performance_report.save_html('production_model_drift_report.html')

# Define the week periods
week1_period = ('2011-01-29 00:00:00', '2011-02-07 23:00:00')
week2_period = ('2011-02-07 00:00:00', '2011-02-14 23:00:00')
week3_period = ('2011-02-15 00:00:00', '2011-02-21 23:00:00')

# Function to generate and save model drift report for a given week
def generate_model_drift_report(week_period, week_name):
    start_date, end_date = week_period
    week_data = raw_data[(raw_data.index >= start_date) & (raw_data.index <= end_date)]
    week_X = week_data[numerical_features + categorical_features]
    week_y = week_data[target]
    week_data['target'] = week_data[target]
    week_data['prediction'] = regressor.predict(week_X)

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference_jan11, current_data=week_data, column_mapping=column_mapping)
    report.save_html(f'{week_name}_model_drift_report.html')

# Generate model drift reports for each week
generate_model_drift_report(week1_period, 'week1')
generate_model_drift_report(week2_period, 'week2')
generate_model_drift_report(week3_period, 'week3')

# Generate target drift report for the worst week (week 3)
target_drift_report = Report(metrics=[TargetDriftPreset()])
week3_data = raw_data[(raw_data.index >= week3_period[0]) & (raw_data.index <= week3_period[1])]
week3_data['target'] = week3_data[target]
week3_data['prediction'] = regressor.predict(week3_data[numerical_features + categorical_features])  # Ensure the prediction column is complete
target_drift_report.run(reference_data=reference_jan11, current_data=week3_data, column_mapping=column_mapping)
target_drift_report.save_html('week3_target_drift_report.html')

# Data drift report for the last week using numerical variables only
data_drift_column_mapping = ColumnMapping()
data_drift_column_mapping.target = target
data_drift_column_mapping.prediction = prediction
data_drift_column_mapping.numerical_features = numerical_features
data_drift_column_mapping.categorical_features = []

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=reference_jan11, current_data=week3_data, column_mapping=data_drift_column_mapping)
data_drift_report.save_html('week3_data_drift_report.html')
