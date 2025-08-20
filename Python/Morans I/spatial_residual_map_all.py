import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Step 1: 加载数据
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")
pred_df = pd.read_csv("Output/7. Predictions/all_predictions_logistic.csv")
y_df = pd.read_csv("Output/5. Feature engineering/y_full.csv")

# Step 2: 合并数据，计算残差
merged = gdf.merge(pred_df, on="LSOA11CD").merge(y_df, on="LSOA11CD")
merged["residual"] = merged["hotspot_10"] - merged["predicted"]

# Step 3: 设置色彩映射（红-白-蓝）
vmin, vmax = -1, 1
cmap = plt.cm.RdBu_r  # Red=高估，Blue=低估，更柔和自然
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Step 4: 绘图
fig, ax = plt.subplots(1, 1, figsize=(14, 9))
merged.plot(
    column="residual",
    cmap=cmap,
    norm=norm,
    linewidth=0.2,
    edgecolor='gray',
    ax=ax
)

# Step 5: 添加 colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # 挂载空值
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Residuals (Actual - Predicted)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Step 6: 图像优化
ax.set_title("Spatial Residual Map: Logistic Regression", fontsize=16)
ax.axis("off")
plt.tight_layout()

# Step 7: 输出
plt.savefig("Output/8. Moran I residuals/spatial_residual_map_all.png", dpi=300)