# Zillow-Home-Value-Prediction-Pipeline
This project builds an end-to-end machine learning pipeline to predict Zillow home value log error using structured property and transaction data. It includes data cleaning, feature engineering, model training, evaluation, visualization, and model persistence.
Features

Combines 2016–2017 Zillow datasets (train and property data)

Handles missing values and encodes categorical features

Generates advanced engineered features (age, ratios, etc.)

Trains and evaluates multiple models:

Ridge Regression

Lasso Regression

ElasticNet

Random Forest Regressor

XGBoost

LightGBM

Automatically selects the best model (highest R²)

Saves plots and model files to the specified local directory
zelo/
│
├── train_2016_v2.csv
├── train_2017.csv
├── properties_2016.csv
├── properties_2017.csv
│
├── model_performance_comparison.png
├── feature_importances_lightgbm.png
├── actual_vs_predicted_logerror.png
├── best_model_<ModelName>.pkl
└── Zillow_Final_Pipeline.py
Model Evaluation Metrics

Each model is evaluated using:

R² Score (Goodness of fit)

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

Results are displayed and saved as visual comparisons in bar charts.

📊 Visual Outputs

model_performance_comparison.png – compares R² scores of all models

feature_importances_lightgbm.png – shows top predictive features

actual_vs_predicted_logerror.png – scatter plot of actual vs predicted log errors

💾 Output Files
All models and plots are saved automatically in your local directory:
C:\Users\kulsum ansari\OneDrive\Documents\datathon\zelo
Tech Stack

Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)

LightGBM, XGBoost

Joblib (model persistence)
Execution Steps

Place all required CSV files in the zelo folder.

Run the Python file:
python Zillow.ipynb
Check the console for progress and evaluation metrics.

View saved plots and .pkl model in the zelo folder.
📥 Loading datasets...
✅ Merged dataset shape: (167888, 61)
⚙️ Feature Engineering...
Train: (131623, 63), Test: (32906, 63), Total features: 63
✅ Ridge | R²=0.0079 | MAE=0.0531 | RMSE=0.0842 | Time=0.2s
✅ Lasso | R²=0.0047 | MAE=0.0531 | RMSE=0.0843 | Time=1.5s
✅ ElasticNet | R²=0.0070 | MAE=0.0531 | RMSE=0.0842 | Time=3.3s
✅ RandomForest | R²=0.0193 | MAE=0.0528 | RMSE=0.0837 | Time=565.8s
✅ XGBoost | R²=0.0260 | MAE=0.0527 | RMSE=0.0834 | Time=18.0s

🚀 Training LightGBM (custom tuned)...
Training until validation scores don't improve for 100 rounds
[100]	train's rmse: 0.0824859	valid's rmse: 0.0835268
[200]	train's rmse: 0.0812908	valid's rmse: 0.0833366
[300]	train's rmse: 0.080348	valid's rmse: 0.0832885
[400]	train's rmse: 0.0795165	valid's rmse: 0.0832635
[500]	train's rmse: 0.0787489	valid's rmse: 0.0832542
Early stopping, best iteration is:
[480]	train's rmse: 0.0789102	valid's rmse: 0.0832448
✅ LightGBM | R²=0.0296 | MAE=0.0526 | RMSE=0.0832

📊 Model Comparison:
           Model       MAE      RMSE        R²
5      LightGBM  0.052570  0.083245  0.029627
4       XGBoost  0.052681  0.083398  0.026049
3  RandomForest  0.052781  0.083688  0.019262
0         Ridge  0.053093  0.084172  0.007887
2    ElasticNet  0.053066  0.084210  0.006999
1         Lasso  0.053123  0.084309  0.004664
✅ Model comparison plot saved in zelo folder.
✅ Feature importance plot saved in zelo folder.
✅ Actual vs Predicted scatter plot saved in zelo folder.

🏆 Best Model: LightGBM | R²=0.0296
💾 Model saved successfully at: C:\Users\kulsum ansari\OneDrive\Documents\datathon\zelo\best_model_LightGBM.pkl

