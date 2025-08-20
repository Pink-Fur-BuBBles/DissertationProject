import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde

# -------------------------------
# Step 1: 读取 LSOA 边界与模型预测值
# -------------------------------
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")
pred = pd.read_csv("Output/7. Predictions/all_predictions_logistic.csv")  # 包含 LSOA11CD 和 predicted 概率

# 合并预测概率
gdf = gdf.merge(pred[["LSOA11CD", "predicted"]], on="LSOA11CD")

# -------------------------------
# Step 2: 获取 LSOA 几何中心坐标
# -------------------------------
gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y

# -------------------------------
# Step 3: KDE 插值处理
# -------------------------------
values = gdf["predicted"].values
positions = np.vstack([gdf["x"], gdf["y"]])
kde = gaussian_kde(positions, weights=values, bw_method=0.15)

xmin, ymin, xmax, ymax = gdf.total_bounds
xgrid, ygrid = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
z = kde(grid_coords).reshape(xgrid.shape)

# KDE 值归一化
z_normalized = z / np.max(z)

# -------------------------------
# Step 4: 绘图输出
# -------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# KDE 热力图
kde_plot = ax.imshow(
    np.rot90(z_normalized),
    extent=[xmin, xmax, ymin, ymax],
    cmap='YlOrRd',  # 更清晰、适合风险视觉表达
    aspect='auto',
    alpha=0.9
)

# 边界叠加
gdf.boundary.plot(ax=ax, linewidth=0.4, color="black", alpha=0.4)

# 标题与图例
ax.set_title("Smoothed Predicted Crime Risk Heatmap (Logistic Regression)", fontsize=15)
ax.axis("off")
cbar = plt.colorbar(kde_plot, ax=ax, shrink=0.6)
cbar.set_label("Normalized Predicted Risk (0–1)", fontsize=12)

# -------------------------------
# Step 5: 保存输出
# -------------------------------
output_path = "Output/10. Visualization analysis"
os.makedirs(output_path, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{output_path}/kde_predicted_hotspots_logistic.png", dpi=400)
plt.show()