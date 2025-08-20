import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# 参数与路径设定
# -------------------------------
output_dir = "Output/7. Predictions"
os.makedirs(output_dir, exist_ok=True)

model_path = "Output/5b. Re-Feature engineering/logistic_regression_model.joblib"
model = joblib.load(model_path)

# -------------------------------
# 加载数据
# -------------------------------
X_test = pd.read_csv("Output/5. Feature engineering/X_test.csv")
y_test = pd.read_csv("Output/5. Feature engineering/y_test.csv")

X_full = pd.read_csv("Output/5. Feature engineering/X_full.csv")
y_full = pd.read_csv("Output/5. Feature engineering/y_full.csv")

# 确保匹配特征列
feature_cols = ['ptal_score', 'imd_score', 'log_pop_density']

# -------------------------------
# TEST 数据预测
# -------------------------------
X_test_feat = X_test[feature_cols]
y_test_true = y_test.set_index("LSOA11CD")['hotspot_10']

y_test_prob = model.predict_proba(X_test_feat)[:, 1]
y_test_pred = model.predict(X_test_feat)
y_test_resid = y_test_true - y_test_prob

# 合并输出
df_test_output = X_test[['LSOA11CD']].copy()
df_test_output['actual'] = y_test_true.values
df_test_output['predicted'] = y_test_prob
df_test_output['residual'] = y_test_resid.values

df_test_output.to_csv(f"{output_dir}/test_predictions_logistic.csv", index=False)

# -------------------------------
# ALL 数据预测
# -------------------------------
X_all_feat = X_full[feature_cols]
y_all_true = y_full.set_index("LSOA11CD")['hotspot_10']

y_all_prob = model.predict_proba(X_all_feat)[:, 1]
y_all_pred = model.predict(X_all_feat)
y_all_resid = y_all_true - y_all_prob

# 合并输出
df_all_output = X_full[['LSOA11CD']].copy()
df_all_output['actual'] = y_all_true.values
df_all_output['predicted'] = y_all_prob
df_all_output['residual'] = y_all_resid.values

df_all_output.to_csv(f"{output_dir}/all_predictions_logistic.csv", index=False)

print("✅ Logistic Regression prediction & residuals saved for test and all.")