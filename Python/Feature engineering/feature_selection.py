import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设定路径
output_dir = "Output/5a. Feature selection"
os.makedirs(output_dir, exist_ok=True)

# 读取数据（确保已做特征工程）
df = pd.read_csv("Data_cleared/final_dataset_imputed.csv")
from feature_engineering import apply_feature_engineering
df = apply_feature_engineering(df)

# 选择特征和目标
features = [
    'ptal_score',
    'imd_score',
    'log_pop_density',
    'ptal_imd_interact',
    'road_node_ratio'
]
target = 'hotspot_10'

df_model = df[features + ['MSOA11CD', target]].dropna()

# 空间分层划分 train/test
unique_groups = df_model['MSOA11CD'].unique()
train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

train_df = df_model[df_model['MSOA11CD'].isin(train_groups)].copy()
X_train = train_df[features]
y_train = train_df[target]

# 标准化（仅用于 Logistic Regression）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------
# 训练模型
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced', max_iter=1000, solver='liblinear'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=4, use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )
}

# -------------------------------
# 提取特征重要性
# -------------------------------
importance_df = pd.DataFrame({"Feature": features})

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        importance = np.abs(model.coef_[0])
    else:
        model.fit(X_train, y_train)
        importance = model.feature_importances_

    importance_df[name] = importance

# -------------------------------
# 保存结果与可视化
# -------------------------------
importance_df.set_index("Feature").plot.bar(figsize=(10, 6))
plt.title("Feature Importance Comparison Across Models")
plt.ylabel("Importance (absolute weights or Gini importance)")
plt.xticks(rotation=45)
plt.tight_layout()

# 保存
csv_path = os.path.join(output_dir, "feature_importance_comparison.csv")
fig_path = os.path.join(output_dir, "feature_importance_comparison.png")

importance_df.to_csv(csv_path, index=False)
plt.savefig(fig_path)
plt.close()

print(f"✅ Feature importance saved to:\n- CSV: {csv_path}\n- FIG: {fig_path}")