import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

# 路径设定
output_dir = "Output/5b. Re-Feature engineering"
os.makedirs(output_dir, exist_ok=True)

# STEP 1: 加载数据并应用特征工程
df = pd.read_csv('Data_cleared/final_dataset_imputed.csv')

from feature_engineering import apply_feature_engineering
df = apply_feature_engineering(df)

# 所有可选特征
all_features = [
    'ptal_score',
    'imd_score',
    'log_pop_density',
    'ptal_imd_interact',
    'road_node_ratio'
]

# 基础字段
id_cols = ['LSOA11CD', 'MSOA11CD', 'crime_count', 'hotspot_10']

df_model = df[all_features + id_cols].dropna()

# STEP 2: 空间分层 train/test
unique_groups = df_model['MSOA11CD'].unique()
train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

train_df = df_model[df_model['MSOA11CD'].isin(train_groups)].copy()
test_df = df_model[df_model['MSOA11CD'].isin(test_groups)].copy()

# STEP 3: 定义每个模型使用的特征（来自第一轮评估的 top3 特征）
feature_sets = {
    'Logistic Regression': ['ptal_score', 'imd_score', 'log_pop_density'],
    'Random Forest': ['ptal_imd_interact', 'ptal_score', 'log_pop_density'],
    'XGBoost': ['ptal_imd_interact', 'ptal_score', 'log_pop_density']
}

# STEP 4: 模型构建
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, solver='liblinear'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=4, use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(train_df['hotspot_10'] == 0).sum() / (train_df['hotspot_10'] == 1).sum()
    )
}

results = []

for model_name, model in models.items():
    selected_features = feature_sets[model_name]

    # 构建数据
    X_train = train_df[selected_features]
    y_train = train_df['hotspot_10']
    X_test = test_df[selected_features]
    y_test = test_df['hotspot_10']

    # 模型训练
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 模型评估
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': model_name,
        'Features used': ', '.join(selected_features),
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'AUC': auc
    })

    # 保存模型
    joblib.dump(model, os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_model.joblib"))

    # 保存数据文件（用于可视化与进一步研究）
    X_train.to_csv(os.path.join(output_dir, f"{model_name}_X_train.csv"), index=False)

# STEP 5: 输出汇总结果
results_df = pd.DataFrame(results)
results_df = results_df[['Model', 'Features used', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]

print("=== Feature Selection Model Evaluation ===")
print(results_df.round(4))