import pandas as pd
import os

# 设置相对路径
imd_path = os.path.join('Data_mid', 'Census', 'IMD', 'ID 2019 for London.xlsx')  # 替换为你的实际文件名
imd_df = pd.read_excel(imd_path, sheet_name='IMD 2019')

# 选取需要的字段，并重命名方便后续统一
imd_df = imd_df[[
    'LSOA code (2011)',
    'Index of Multiple Deprivation (IMD) Score',
    'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)'
]].rename(columns={
    'LSOA code (2011)': 'LSOA11CD',
    'Index of Multiple Deprivation (IMD) Score': 'imd_score',
    'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)': 'imd_decile'
})

# 仅保留合法 Decile 范围（1~10）并去除缺失
imd_df = imd_df[imd_df['imd_decile'].between(1, 10)]
imd_df = imd_df.dropna(subset=['imd_score', 'imd_decile'])

# 输出为 CSV
output_path = os.path.join('Data_cleared', 'imd_cleaned.csv')
imd_df.to_csv(output_path, index=False)