import pandas as pd
import os

# 1. 路径设置
pop_path = os.path.join('Data_mid', 'Census', 'Population density', '301978379403041.csv')  # 替换为你的实际路径

# 2. 读取数据，无列名
df = pd.read_csv(pop_path, header=None, names=['lsoa_raw', 'pop_density'])

# 3. 拆分 'E01000027 : Name' 为 LSOA11CD 和 LSOA 名称
df[['LSOA11CD', 'LSOA_Name']] = df['lsoa_raw'].str.split(' : ', expand=True)

# 4. 保留需要的两列
df_cleaned = df[['LSOA11CD', 'pop_density']].copy()

# 5. 转换人口密度为 float
df_cleaned['pop_density'] = pd.to_numeric(df_cleaned['pop_density'], errors='coerce')

# 6. 去除空值
df_cleaned.dropna(subset=['pop_density'], inplace=True)

# 7. 导出为标准 CSV
output_path = os.path.join('Data_cleared', 'population_density_cleaned.csv')
df_cleaned.to_csv(output_path, index=False)