import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -------------------------------
# STEP 1: 加载模型与数据
# -------------------------------
model_path = "Output/5b. Re-Feature engineering/logistic_regression_model.joblib"
data_path = "Output/5b. Re-Feature engineering/Logistic Regression_X_train.csv"

model = joblib.load(model_path)
X_train = pd.read_csv(data_path)

# -------------------------------
# STEP 2: 创建 SHAP Explainer
# -------------------------------
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# -------------------------------
# STEP 3: 输出 SHAP Summary Plot（散点）
# -------------------------------
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP Summary Plot (Logistic Regression)")
plt.tight_layout()
plt.savefig("Output/6. SHAP/shap_summary_logistic.png", dpi=300)
plt.close()

# -------------------------------
# STEP 4: 输出 SHAP Bar Plot（平均重要性）
# -------------------------------
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Bar Plot (Logistic Regression)")
plt.tight_layout()
plt.savefig("Output/6. SHAP/shap_bar_logistic.png", dpi=300)
plt.close()

print("✅ SHAP summary 和 bar plot 已保存至 Output/6. SHAP/")

# -------------------------------
# STEP 5: Force Plot（局部解释，选一个样本）
# -------------------------------
sample_idx = 100  # 可自行调整
force_plot_path = f"Output/6. SHAP/shap_force_logistic_{sample_idx}.html"

shap.save_html(force_plot_path, shap.plots.force(shap_values[sample_idx]))