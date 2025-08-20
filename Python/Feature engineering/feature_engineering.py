import numpy as np

def apply_feature_engineering(df):
    required_columns = ['pop_density', 'ptal_score', 'imd_score', 'road_density', 'node_density']
    assert all(col in df.columns for col in required_columns), "缺少必要字段，请检查列名。"

    # 特征 1：人口密度对数变换
    df['log_pop_density'] = np.log1p(df['pop_density'])

    # 特征 2：PTAL × IMD 交互项
    df['ptal_imd_interact'] = df['ptal_score'] * df['imd_score']

    # 特征 3：道路/节点密度比（避免除以 0）
    df['road_node_ratio'] = df['road_density'] / (df['node_density'] + 1)

    return df