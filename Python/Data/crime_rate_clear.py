import pandas as pd
import os

# === 1. 设置相对路径读取文件 ===
relative_path = os.path.join('Data_mid', 'Crime rate', 'MPS LSOA Level Crime (Historical).csv')  # 你可调整为你实际的路径
df = pd.read_csv(relative_path)

# === 2. 选取时间列（2021.01–2023.12） ===
date_cols = [col for col in df.columns if col.isdigit() and 202001 <= int(col) <= 202212]
df.rename(columns={'LSOA Code': 'LSOA11CD'}, inplace=True)
df = df[['LSOA11CD'] + date_cols]

# === 3. 聚合犯罪记录：同一LSOA多种犯罪类型求和 ===
crime_agg = df.groupby('LSOA11CD')[date_cols].sum()

# === 4. 总犯罪数（2021–2023） ===
crime_agg['crime_count'] = crime_agg.sum(axis=1)

# === 5. 构建多个 hotspot 标签 ===
for q in [0.05, 0.10, 0.15, 0.20]:
    threshold = crime_agg['crime_count'].quantile(1 - q)
    label_col = f"hotspot_{int(q * 100)}"
    crime_agg[label_col] = (crime_agg['crime_count'] >= threshold).astype(int)

# === 6. 重置索引并保存 ===
crime_agg.reset_index(inplace=True)
output_path = os.path.join('Data_cleared', 'crime_cleaned.csv')
crime_agg.to_csv(output_path, index=False)