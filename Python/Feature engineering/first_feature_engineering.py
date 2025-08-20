import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1: 读取数据并应用特征工程
# -------------------------------
df = pd.read_csv('Data_cleared/final_dataset_imputed.csv')

from feature_engineering import apply_feature_engineering
df = apply_feature_engineering(df)

# 特征列：来自你定义的特征工程模块输出
feature_cols = [
    'ptal_score',
    'imd_score',
    'log_pop_density',
    'ptal_imd_interact',
    'road_node_ratio'
]

base_cols = ['LSOA11CD', 'MSOA11CD', 'crime_count', 'hotspot_10']
df_model = df[feature_cols + base_cols].dropna()

# -------------------------------
# STEP 2: 空间分层划分 Train/Test
# -------------------------------
unique_groups = df_model['MSOA11CD'].unique()
train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

train_df = df_model[df_model['MSOA11CD'].isin(train_groups)].copy()
test_df = df_model[df_model['MSOA11CD'].isin(test_groups)].copy()

X_train = train_df[feature_cols]
y_train = train_df['hotspot_10']
X_test = test_df[feature_cols]
y_test = test_df['hotspot_10']

# -------------------------------
# STEP 3: 模型训练与评估
# -------------------------------
output_dir = "Output/5. Feature engineering"
os.makedirs(output_dir, exist_ok=True)

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

results = []
plt.figure(figsize=(8, 6))

# 初始化模型引用字典（训练后用于保存）
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    # 保存模型对象
    trained_models[name] = model

# -------------------------------
# STEP 4: 保存 ROC 图
# -------------------------------
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(output_dir, "roc_curve.png")
plt.savefig(roc_path)
plt.close()
print(f"ROC Curve saved to: {roc_path}")

# -------------------------------
# STEP 5: 输出模型评估结果
# -------------------------------
results_df = pd.DataFrame(results)
print("Model performance evaluation results:\n")
print(results_df.round(4))

# -------------------------------
# STEP 6: 保存模型和训练集
# -------------------------------
joblib.dump(trained_models["Logistic Regression"], os.path.join(output_dir, "logistic_regression_model.joblib"))
joblib.dump(trained_models["Random Forest"], os.path.join(output_dir, "random_forest_model.joblib"))
joblib.dump(trained_models["XGBoost"], os.path.join(output_dir, "xgboost_model.joblib"))

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)

# -------------------------------
# STEP 7: 保存可视化数据（全体和测试集）
# -------------------------------
# 保存测试集用于预测评估图
test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
test_df[['hotspot_10', 'LSOA11CD']].to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

# 保存 full dataset 供预测输出与可视化
df_model.to_csv(os.path.join(output_dir, "X_full.csv"), index=False)
df_model[['hotspot_10', 'LSOA11CD']].to_csv(os.path.join(output_dir, "y_full.csv"), index=False)