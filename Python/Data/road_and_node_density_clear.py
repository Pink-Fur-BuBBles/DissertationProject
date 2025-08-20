import pandas as pd
import os

# 1. 设置路径
road_node_path = os.path.join('Data_mid', 'Road Density And Intersection Density', 'road_and_node_density.csv')  # 替换你的路径

# 2. 读取原始 CSV
df = pd.read_csv(road_node_path)

# 3. 重命名（可选，统一命名规范）
df.rename(columns={
    'road_densi': 'road_density',
    'node_densi': 'node_density'
}, inplace=True)

# 4. 类型转换（确保数值型）
df['road_density'] = pd.to_numeric(df['road_density'], errors='coerce')
df['node_density'] = pd.to_numeric(df['node_density'], errors='coerce')

# 5. 提取核心列
df_cleaned = df[['LSOA11CD', 'road_density', 'node_density']].copy()

# 6. 删除缺失值（如有）
df_cleaned.dropna(subset=['road_density', 'node_density'], inplace=True)

# 7. 保存为处理后文件
output_path = os.path.join('Data_cleared', 'road_node_density_cleaned.csv')
df_cleaned.to_csv(output_path, index=False)