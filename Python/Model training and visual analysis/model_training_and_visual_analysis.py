import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
import os

# -------------------------------
# 读取并准备数据
# -------------------------------
df = pd.read_csv('Data_cleared/final_dataset_imputed.csv')

features = ['ptal_score', 'imd_score', 'pop_density', 'road_density', 'node_density']
base_cols = ['LSOA11CD', 'MSOA11CD', 'crime_count', 'hotspot_10']
df_model = df[features + base_cols].dropna()

# 空间分层：MSOA级抽样
unique_groups = df_model['MSOA11CD'].unique()
train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)
train_df = df_model[df_model['MSOA11CD'].isin(train_groups)].copy()
test_df = df_model[df_model['MSOA11CD'].isin(test_groups)].copy()

# -------------------------------
# 构建训练和测试输入输出
# -------------------------------
X_train = train_df[features]
y_train = train_df['hotspot_10']
X_test = test_df[features]
y_test = test_df['hotspot_10']

# 构建完整数据集（用于全图预测）
X_full = df_model[features]
y_full = df_model[['hotspot_10', 'LSOA11CD', 'crime_count'] + features]

# -------------------------------
# 模型训练与评估
# -------------------------------
output_dir = "Output/3. Model training and visual analysis"
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

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# -------------------------------
# 保存数据集（用于可视化）
# -------------------------------
test_out = test_df[['LSOA11CD', 'crime_count', 'hotspot_10'] + features]
full_out = df_model[['LSOA11CD', 'crime_count', 'hotspot_10'] + features]

test_out.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
test_out[['hotspot_10', 'LSOA11CD']].to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

full_out.to_csv(os.path.join(output_dir, "X_full.csv"), index=False)
full_out[['hotspot_10', 'LSOA11CD']].to_csv(os.path.join(output_dir, "y_full.csv"), index=False)

# -------------------------------
# 输出模型评估表格
# -------------------------------
results_df = pd.DataFrame(results)
print("Model performance evaluation results:\n")
print(results_df.round(4))