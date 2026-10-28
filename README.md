# Food-Delivery-Time-Predection

Predict how many minutes a food delivery will take, given the order details — who is delivering it, how far it has to travel, the weather, traffic, time of day, and more. Framed as a supervised regression problem and solved end-to-end, from raw messy data to a saved model ready for inference.


# Approach

Load & inspect the raw data
Clean messy text and missing values (e.g. "conditions Sunny" → Sunny, "(min) 24" → 24)
EDA — find which features actually move delivery time
Feature engineering
distance_km from restaurant/customer coordinates via the Haversine formula
order_hour extracted from the order timestamp
Encode categoricals (ordinal for traffic density, one-hot for the rest) and scale features
Train & compare Linear Regression, Decision Tree, Random Forest, XGBoost
Tune the best model (RandomizedSearchCV), evaluate (RMSE, R², Adjusted R²), and save it for inference
The maths and reasoning behind each step are explained inline in the notebook.


# Tech stack

thon · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · joblib
