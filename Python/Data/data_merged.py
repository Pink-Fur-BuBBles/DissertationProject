import pandas as pd
import os

# ✅ 设置基础路径
base_path = 'Data_cleared'

# ✅ 读取主控 LSOA 列表（用来保留所有合法 LSOA11CD）
lsoa_df = pd.read_csv(os.path.join(base_path, 'London_lsoa.csv'), dtype={'LSOA11CD': str})

# ✅ 加载清洗后的各变量数据
crime_df = pd.read_csv(os.path.join(base_path, 'crime_cleaned.csv'))             # LSOA11CD, crime_count, hotspot_5, hotspot_10, hotspot_15, hotspot_20
imd_df   = pd.read_csv(os.path.join(base_path, 'imd_cleaned.csv'))               # LSOA11CD, imd_score, imd_decile
ptal_df  = pd.read_csv(os.path.join(base_path, 'ptal_cleaned.csv'))              # LSOA11CD, ptal
pop_df   = pd.read_csv(os.path.join(base_path, 'population_density_cleaned.csv'))       # LSOA11CD, pop_density
road_df  = pd.read_csv(os.path.join(base_path, 'road_node_density_cleaned.csv'))      # LSOA11CD, road_density, node_density

# ✅ 用 LSOA 主控表做主键，依次左连接各变量
df = lsoa_df[['LSOA11CD']].copy()

df = df.merge(crime_df, on='LSOA11CD', how='left')
df = df.merge(imd_df, on='LSOA11CD', how='left')
df = df.merge(ptal_df, on='LSOA11CD', how='left')
df = df.merge(pop_df, on='LSOA11CD', how='left')
df = df.merge(road_df, on='LSOA11CD', how='left')

# ✅ 精简字段列表
columns_to_keep = [
    'LSOA11CD', 'crime_count', 'hotspot_5', 'hotspot_10', 'hotspot_15', 'hotspot_20', 
    'imd_score', 'imd_decile',
    'ptal_score', 'pop_density', 'road_density', 'node_density'
]

# ✅ 删除不必要列，仅保留分析变量
df_cleaned = df[columns_to_keep]

# ✅ 输出最终精简数据集
output_path_cleaned = os.path.join(base_path, 'merged_data_with_hotspot_sensitivity.csv')
df_cleaned.to_csv(output_path_cleaned, index=False)

print(f"\n✅ 已输出精简版数据集（用于建模）：{output_path_cleaned}")