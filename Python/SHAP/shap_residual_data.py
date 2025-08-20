import pandas as pd
import joblib
import shap
import os

# 路径设定
model_path = "Output/5b. Re-Feature engineering/logistic_regression_model.joblib"
X_path = "Output/5. Feature engineering/X_full.csv"
y_path = "Output/5. Feature engineering/y_full.csv"
output_dir = "Output/6. SHAP"
os.makedirs(output_dir, exist_ok=True)

# 加载模型和数据
model = joblib.load(model_path)
X = pd.read_csv(X_path)
y = pd.read_csv(y_path)

# -------------------------------
# STEP 1: 明确特征列
# -------------------------------
feature_cols = ['ptal_score', 'imd_score', 'log_pop_density']

# -------------------------------
# STEP 2: 加载并筛选数据
# -------------------------------
X_raw = pd.read_csv("Output/5. Feature engineering/X_full.csv")
X = X_raw[feature_cols]  # ✅ 只保留训练时使用的特征列

y = pd.read_csv("Output/5. Feature engineering/y_full.csv")
lsoa_codes = y['LSOA11CD'].values
y_true = y['hotspot_10'].values

# -------------------------------
# STEP 3: 模型预测 & 残差计算
# -------------------------------
y_pred_proba = model.predict_proba(X)[:, 1]
residuals = y_pred_proba - y_true

# -------------------------------
# STEP 4: SHAP 值计算
# -------------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap_df = pd.DataFrame(shap_values.values, columns=[f"shap_{col}" for col in X.columns])

# -------------------------------
# STEP 5: 合并输出保存
# -------------------------------
output_df = pd.concat([X.reset_index(drop=True), shap_df], axis=1)
output_df["LSOA11CD"] = lsoa_codes
output_df["actual"] = y_true
output_df["predicted"] = y_pred_proba
output_df["residual"] = residuals

# 保存 CSV
output_path = "Output/6. SHAP/shap_residual_data.csv"
output_df.to_csv(output_path, index=False)