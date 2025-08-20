import pandas as pd
import os

# 1. 读取 PTAL 原始数据（路径示例）
ptal_path = os.path.join('Data_mid', 'POIs-Extra', 'PTAL', 'LSOA2011 AvPTAI2015.csv')
ptal = pd.read_csv(ptal_path)

# 2. 保留关键信息：LSOA 代码与平均指数
ptal_cleaned = ptal[['LSOA2011', 'AvPTAI2015']].copy()

# 3. 重命名列，统一格式
ptal_cleaned.columns = ['LSOA11CD', 'ptal_score']

# 4. 去重检查（如果有重复 LSOA）
ptal_cleaned = ptal_cleaned.drop_duplicates(subset='LSOA11CD')

# 5. 保存清洗后的数据
output_path = os.path.join('Data_cleared', 'ptal_cleaned.csv')
ptal_cleaned.to_csv(output_path, index=False)