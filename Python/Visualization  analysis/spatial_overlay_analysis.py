import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Step 1: 读取 LSOA shapefile + 模型预测
# -------------------------------
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")
pred_df = pd.read_csv("Output/7. Predictions/test_predictions_logistic.csv")  # 包含 LSOA11CD, predicted
crime_df = pd.read_csv("Data_cleared/final_dataset_imputed.csv")[["LSOA11CD", "crime_count"]]

# -------------------------------
# Step 2: KDE 计算（简化版：每个 LSOA 的 crime_count 密度）
# -------------------------------
# 按 LSOA 汇总总量
merged = gdf.merge(crime_df, on="LSOA11CD", how="left")
merged["area_km2"] = merged.geometry.to_crs(epsg=3857).area / 1e6
merged["crime_density"] = merged["crime_count"] / merged["area_km2"]

# KDE 替代变量（可近似代表 KDE）
kde_df = merged[["LSOA11CD", "crime_density"]]

# -------------------------------
# Step 3: 合并 predicted + KDE proxy
# -------------------------------
merged_pred = pred_df.merge(kde_df, on="LSOA11CD")

# 缩放两者到 0–1，便于可视化
scaler = MinMaxScaler()
merged_pred[["crime_density_scaled", "predicted_scaled"]] = scaler.fit_transform(
    merged_pred[["crime_density", "predicted"]]
)

# -------------------------------
# Step 4: 可视化 + 计算相关性
# -------------------------------
plt.figure(figsize=(8, 6))
sns.regplot(
    x="crime_density_scaled",
    y="predicted_scaled",
    data=merged_pred,
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "red"}
)
plt.xlabel("KDE-based Crime Density (scaled)")
plt.ylabel("Model Predicted Probability (scaled)")
plt.title("Correlation between KDE and Model Prediction")
plt.tight_layout()
plt.savefig("Output/10. Visualization  analysis/kde_prediction_correlation.png", dpi=300)

# -------------------------------
# Step 5: Pearson 相关系数
# -------------------------------
r, p = pearsonr(merged_pred["crime_density"], merged_pred["predicted"])
print(f"Pearson Correlation (unscaled): r = {r:.3f}, p = {p:.4f}")