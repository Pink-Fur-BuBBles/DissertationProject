import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import os

# -------------------------------
# Step 1: 加载 LSOA 边界 & 数据
# -------------------------------
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")

# crime_count & predicted 数据合并
df = pd.read_csv("Data_cleared/final_dataset_imputed.csv")[["LSOA11CD", "crime_count"]]
pred = pd.read_csv("Output/7. Predictions/all_predictions_logistic.csv")[["LSOA11CD", "predicted"]]
df = df.merge(pred, on="LSOA11CD", how="left")
gdf = gdf.merge(df, on="LSOA11CD", how="left")

# -------------------------------
# Step 2: 获取 LSOA 几何中心坐标
# -------------------------------
gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y

# -------------------------------
# Step 3: KDE 生成实际值 z_actual
# -------------------------------
positions = np.vstack([gdf["x"], gdf["y"]])

kde_actual = gaussian_kde(positions, weights=gdf["crime_count"], bw_method=0.15)
kde_pred = gaussian_kde(positions, weights=gdf["predicted"], bw_method=0.15)

# 创建网格
xmin, ymin, xmax, ymax = gdf.total_bounds
xgrid, ygrid = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])

z_actual = kde_actual(grid_coords).reshape(xgrid.shape)
z_predicted = kde_pred(grid_coords).reshape(xgrid.shape)

# -------------------------------
# Step 4: 计算残差
# -------------------------------
residual_z = z_predicted - z_actual
vmax = np.max(np.abs(residual_z))
vmin = -vmax

# -------------------------------
# Step 5: 绘图
# -------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

resid_plot = ax.imshow(
    np.rot90(residual_z),
    extent=[xmin, xmax, ymin, ymax],
    cmap="bwr",  # 蓝白红
    vmin=vmin,
    vmax=vmax,
    alpha=0.9
)

# 添加 LSOA 边界
gdf.boundary.plot(ax=ax, linewidth=0.4, color="black", alpha=0.5)

# 图像美化
ax.set_title("Smoothed Residual Map (Model - Actual KDE)", fontsize=15)
ax.axis("off")

# 添加 colorbar
cbar = plt.colorbar(resid_plot, ax=ax, shrink=0.6)
cbar.set_label("Smoothed Residual (Predicted - Actual)", fontsize=12)

# -------------------------------
# Step 6: 保存输出
# -------------------------------
output_path = "Output/10. Visualization  analysis"
os.makedirs(output_path, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{output_path}/kde_residual_map.png", dpi=400)