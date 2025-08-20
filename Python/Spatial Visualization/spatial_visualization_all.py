import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# -------------------------------
# 读取边界与预测数据
# -------------------------------
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")
gdf = gdf.set_index("LSOA11CD")

df = pd.read_csv("Output/7. Predictions/all_predictions_logistic.csv")
gdf = gdf.merge(df, on="LSOA11CD", how="left")

# -------------------------------
# 创建图像窗口
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# -------------------------------
# 1. 实际 Hotspot 图（0-1）
# 使用明确定义的类别颜色
# -------------------------------
cmap_actual = colors.ListedColormap(["#f0f0f0", "#d73027"])  # 灰白底 + 红色热点
bounds = [-0.1, 0.5, 1.1]
norm_actual = colors.BoundaryNorm(bounds, cmap_actual.N)

gdf.plot(
    column="actual",
    cmap=cmap_actual,
    norm=norm_actual,
    legend=True,
    ax=axes[0],
    edgecolor='lightgrey',
    linewidth=0.2
)
axes[0].set_title("Actual Crime Hotspots (Top 10%)", fontsize=14)
axes[0].axis('off')

# -------------------------------
# 2. 预测概率图（连续色带）
# 使用更清晰渐变色 + alpha 增强可视性
# -------------------------------
gdf.plot(
    column="predicted",
    cmap='YlOrRd',
    legend=True,
    ax=axes[1],
    edgecolor='lightgrey',
    linewidth=0.2,
    alpha=0.95,
    vmin=0,
    vmax=1
)
axes[1].set_title("Predicted Crime Risk", fontsize=14)
axes[1].axis('off')

# -------------------------------
# 输出图像
# -------------------------------
plt.tight_layout()
plt.savefig("Output/9. Spatial Visualization/all_hvp.png", dpi=300)
plt.close()