import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

thresholds = [5, 10, 15, 20]
results = []

df = pd.read_csv('Data_cleared/final_dataset_imputed.csv')

# 特征列
features = ['ptal_score', 'imd_score', 'pop_density', 'road_density', 'node_density']

for threshold in thresholds:
    label_col = f"hotspot_{threshold}"
    
    df_model = df[features + [label_col, 'MSOA11CD']].dropna()
    
    # 空间分组划分（同上）
    groups = df_model['MSOA11CD'].unique()
    train_groups, test_groups = train_test_split(groups, test_size=0.2, random_state=42)
    
    train_df = df_model[df_model['MSOA11CD'].isin(train_groups)]
    test_df = df_model[df_model['MSOA11CD'].isin(test_groups)]
    
    X_train, y_train = train_df[features], train_df[label_col]
    X_test, y_test = test_df[features], test_df[label_col]

    # XGBoost 训练
    model = XGBClassifier(
        n_estimators=100, max_depth=4,
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results.append({
        'Hotspot_Threshold(%)': threshold,
        'AUC': auc,
        'F1': f1,
        'Precision': prec,
        'Recall': rec
    })

# 转换为 DataFrame 并展示
results_df = pd.DataFrame(results)
print(results_df)

# 可视化 AUC vs Threshold 曲线
plt.figure(figsize=(7, 5))
plt.plot(results_df['Hotspot_Threshold(%)'], results_df['AUC'], marker='o')
plt.title("AUC vs Hotspot Threshold")
plt.xlabel("Hotspot Threshold (%)")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
plt.savefig("Output/4. Sensitivity analysis/auc_vs_threshold.png")
plt.close()