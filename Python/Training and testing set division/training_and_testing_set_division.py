import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1: 读取数据
# -------------------------------
df = pd.read_csv('Data_cleared/final_dataset_imputed.csv')

# -------------------------------
# STEP 2: 保留所需字段
# -------------------------------
features = [
    'ptal_score', 'imd_score', 'pop_density',
    'road_density', 'node_density'
]
target = 'hotspot_10'
group = 'MSOA11CD'
id_col = 'LSOA11CD'

# 必须保留 LSOA11CD 用于后续空间合并
df_model = df[features + [target, group, id_col]].dropna()

# -------------------------------
# STEP 3: 进行空间分层抽样
# -------------------------------
# 方法：按 MSOA 分组，随机选择 80% MSOA 为训练集
unique_groups = df_model[group].unique()

train_groups, test_groups = train_test_split(
    unique_groups,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# STEP 4: 基于 MSOA 分配样本
# -------------------------------
train_df = df_model[df_model[group].isin(train_groups)].copy()
test_df = df_model[df_model[group].isin(test_groups)].copy()

# -------------------------------
# STEP 5: 提取训练/测试输入输出
# -------------------------------
X_train = train_df[features + [id_col]]
y_train = train_df[[target, id_col]]

X_test = test_df[features + [id_col]]
y_test = test_df[[target, id_col]]

# -------------------------------
# STEP 6: 构建全体数据（用于全量模型训练 & 空间预测）
# -------------------------------
X_full = df_model[features + [id_col]].copy()
y_full = df_model[[target, id_col]].copy()

# 可选保存为 CSV
X_full.to_csv("Output/2. Training and testing set division/X_full.csv", index=False)
y_full.to_csv("Output/2. Training and testing set division/y_full.csv", index=False)

# -------------------------------
# OPTIONAL: 检查分布平衡
# -------------------------------
# 计算 hotspot 比例（只针对标签列）
train_ratio = y_train[target].mean()
test_ratio = y_test[target].mean()

print("Train set hotspot ratio:", train_ratio)
print("Test set hotspot ratio:", test_ratio)

# -------------------------------
# 输出结果大小
# -------------------------------
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")