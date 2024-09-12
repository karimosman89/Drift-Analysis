# Drift-Analysis
# Bike Sharing Data Drift Analysis

## Overview
This project performs data drift analysis on the Bike Sharing dataset using the Evidently AI library. The analysis includes model validation, production model drift reports, and specific drift analysis for different weeks.

## Steps
1. **Model Validation Report**: Validates the RandomForestRegressor model using January data.
2. **Production Model Drift Report**: Analyzes the drift on the whole January dataset.
3. **Weekly Model Drift Reports**: Generates drift reports for weeks 1, 2, and 3 of February.
4. **Target Drift Report**: Analyzes the target drift for the week with the highest observed drift.
5. **Data Drift Report**: Analyzes data drift for the last week of the observed period.

## Questions and Answers

### After step 4, explain what changed over weeks 1, 2, and 3.
- Over the weeks, the demand patterns for bike rentals likely fluctuated due to varying weather conditions, holidays, and weekdays. These changes in demand might have affected the model's prediction accuracy.

### After step 5, explain what seems to be the root cause of the drift (only using data)?
- The root cause of the drift observed in the target drift report appears to be due to significant changes in weather conditions (temperature, humidity, windspeed) and possibly due to changes in user behavior on different days of the week or specific events during February.

### After step 6, explain what strategy to apply.
- To handle data drift, we should regularly update the model with recent data and possibly implement a dynamic model retraining schedule. Additionally, including more relevant features or using more advanced drift detection and adaptation techniques could help mitigate the impact of drift.

## Single Command to Execute the Script
```sh
python bike_sharing_monitoring.py
 
