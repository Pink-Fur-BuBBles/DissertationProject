import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen, KNN
from esda.moran import Moran
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 1. 读取带 geometry 的 GeoDataFrame（包括 crime_count 和模型输出）
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")
gdf = gdf.set_index("LSOA11CD")

# 读取模型结果
df = pd.read_csv("Output/7. Predictions/test_predictions_logistic.csv")

# 合并
gdf = gdf.merge(df, on="LSOA11CD", how="left")

# ✅ 2. 构造空间权重矩阵（KNN 邻接）
w = KNN.from_dataframe(gdf, k=5)
w.transform = 'r'

# ✅ 3. 检验指标列表
indicators = {
    "Logistic Regression": "predicted",
    "Residuals": "residual"  # residual = true - predicted
}

# ✅ 4. 绘图初始化
fig, axes = plt.subplots(1, len(indicators), figsize=(7 * len(indicators), 5))
sns.set(style="whitegrid")

# ✅ 5. 循环执行 Moran’s I 检验
for i, (title, column) in enumerate(indicators.items()):
    x = gdf["predicted"].dropna()
    gdf_nonan = gdf[gdf["predicted"].notna()]
    w_subset = KNN.from_dataframe(gdf_nonan, k=5)
    w_subset.transform = 'r'

    moran = Moran(x, w_subset)

    axes[i].set_title(f"{title}\nMoran's I = {moran.I:.3f}, p = {moran.p_sim:.4f}")
    axes[i].hist(moran.sim, bins=30, color='skyblue', edgecolor='white')
    axes[i].axvline(moran.I, color='red', linestyle='--', linewidth=2, label="Observed I")
    axes[i].legend()

plt.tight_layout()
plt.savefig("Output/8. Moran I residuals/morans_I_comparison_test.png", dpi=300)
plt.close()