import pandas as pd
import os

# === 设置路径 ===
base_path = 'Data_cleared'

# === 读取数据 ===
df = pd.read_csv(os.path.join(base_path, 'merged_data_with_hotspot_sensitivity.csv'), dtype={'LSOA11CD': str})
lsoa_master = pd.read_csv(os.path.join(base_path, 'London_lsoa.csv'), dtype={'LSOA11CD': str, 'MSOA11CD': str, 'RGN11CD': str})

# === 合并编码信息（MSOA + Region） ===
if 'MSOA11CD' not in df.columns or 'RGN11CD' not in df.columns:
    df = df.merge(lsoa_master[['LSOA11CD', 'MSOA11CD', 'RGN11CD']], on='LSOA11CD', how='left')

# === 要填补的字段及精度设定 ===
vars_to_impute = [
    'crime_count', 'hotspot_5', 'hotspot_10', 'hotspot_15', 'hotspot_20',
    'ptal_score', 'pop_density', 'road_density', 'node_density'
]
decimal_rounding = {
    'crime_count': 0,
    'hotspot_5': 0,
    'hotspot_10': 0,
    'hotspot_15': 0,
    'hotspot_20': 0,
    'ptal_score': 3,
    'pop_density': 1,
    'road_density': 1,
    'node_density': 1
}

# === 计算均值：MSOA 和 Region 层级 ===
msoa_means = df.groupby('MSOA11CD')[vars_to_impute].mean().reset_index()
region_means = df.groupby('RGN11CD')[vars_to_impute].mean().reset_index()

# === 依次填补字段 ===
for var in vars_to_impute:
    # 合并 MSOA 均值
    df = df.merge(msoa_means[['MSOA11CD', var]], on='MSOA11CD', how='left', suffixes=('', '_msoa'))
    # 填补缺失
    df[var] = df[var].fillna(df[f'{var}_msoa'])
    df.drop(columns=[f'{var}_msoa'], inplace=True)

    # 合并 RGN 均值兜底
    df = df.merge(region_means[['RGN11CD', var]], on='RGN11CD', how='left', suffixes=('', '_region'))
    df[var] = df[var].fillna(df[f'{var}_region'])
    df.drop(columns=[f'{var}_region'], inplace=True)

    # 精度处理
    if decimal_rounding.get(var) is not None:
        df[var] = df[var].round(decimal_rounding[var])

# === 输出最终文件 ===
output_path = os.path.join(base_path, 'final_dataset_imputed.csv')
df.to_csv(output_path, index=False)

print(f"\n✅ 三层填补完成，输出文件：{output_path}")