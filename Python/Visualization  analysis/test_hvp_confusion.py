import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.metrics import roc_curve

# ------------------------------
# Step 1: 读取数据
# ------------------------------
gdf = gpd.read_file("Data_mid/Auxiliary boundary data/LSOA/LSOA_2011_London_gen_MHW.shp")
y_test = pd.read_csv("Output/5. Feature engineering/y_test.csv")
pred_df = pd.read_csv("Output/7. Predictions/test_predictions_logistic.csv")

# ------------------------------
# Step 2: 计算最佳阈值
# ------------------------------
fpr, tpr, thresholds = roc_curve(y_test["hotspot_10"], pred_df["predicted"])
j_scores = tpr - fpr
optimal_idx = j_scores.argmax()
optimal_threshold = thresholds[optimal_idx]
print(f"\n✅ Optimal threshold (Youden's J): {optimal_threshold:.3f}")

# ------------------------------
# Step 3: 应用阈值生成分类标签
# ------------------------------
pred_df["predicted_label"] = (pred_df["predicted"] >= optimal_threshold).astype(int)

# ------------------------------
# Step 4: 合并数据
# ------------------------------
merged = gdf.merge(y_test, on="LSOA11CD").merge(pred_df[["LSOA11CD", "predicted", "predicted_label"]], on="LSOA11CD")

# ------------------------------
# Step 5: 分类
# ------------------------------
def classify(row):
    if row["hotspot_10"] == 1 and row["predicted_label"] == 1:
        return "True Positive"
    elif row["hotspot_10"] == 0 and row["predicted_label"] == 1:
        return "False Positive"
    elif row["hotspot_10"] == 1 and row["predicted_label"] == 0:
        return "False Negative"
    else:
        return "True Negative"

merged["confusion_label"] = merged.apply(classify, axis=1)

# ------------------------------
# Step 6: 更优配色方案（ColorBrewer）
# ------------------------------
color_dict = {
    "True Positive": "#1b9e77",     # teal
    "False Positive": "#d95f02",    # orange
    "False Negative": "#7570b3",    # purple
    "True Negative": "#e6e6e6"      # light gray
}
merged["color"] = merged["confusion_label"].map(color_dict)

# ------------------------------
# Step 7: 绘图
# ------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

for label, color in color_dict.items():
    merged[merged["confusion_label"] == label].plot(ax=ax, color=color, edgecolor='black', linewidth=0.2)

ax.set_title("Confusion Map (Test Set) - Logistic Regression", fontsize=16)
ax.axis("off")

# ------------------------------
# Step 8: 自定义图例
# ------------------------------
legend_patches = [
    mpatches.Patch(color=color_dict["True Positive"], label="True Positive"),
    mpatches.Patch(color=color_dict["False Positive"], label="False Positive"),
    mpatches.Patch(color=color_dict["False Negative"], label="False Negative"),
    mpatches.Patch(color=color_dict["True Negative"], label="True Negative")
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=10, title="Prediction Outcome", title_fontsize=11)

plt.tight_layout()

# ------------------------------
# Step 9: 保存图像
# ------------------------------
output_dir = "Output/10. Visualization  analysis"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/confusion_map_logistic_test.png", dpi=300)

# ------------------------------
# Step 10: 输出统计汇总表
# ------------------------------
summary = merged["confusion_label"].value_counts().reset_index()
summary.columns = ["Category", "Count"]
summary["Percentage"] = (summary["Count"] / summary["Count"].sum()) * 100

print("\nConfusion Category Summary:")
print(summary.to_string(index=False, float_format="%.2f"))