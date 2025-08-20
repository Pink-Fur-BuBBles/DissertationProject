import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# 路径设置（请根据实际路径调整）
data_path = "Output/6. SHAP/shap_residual_data.csv"  # 示例输入
output_path = "Output/10. Visualization  analysis"
os.makedirs(output_path, exist_ok=True)

# 加载数据
df = pd.read_csv(data_path)

# 设置 SHAP 列名前缀
shap_cols = [col for col in df.columns if col.startswith('shap_')]

# 遍历每个 SHAP 特征，绘图和相关性计算
for shap_col in shap_cols:
    feature = shap_col.replace("shap_", "")  # 去掉前缀恢复原始特征名
    
    # 计算皮尔森相关
    r, p = pearsonr(df[shap_col], df["residual"])
    
    # 可视化
    plt.figure(figsize=(7, 5))
    sns.regplot(x=df[shap_col], y=df["residual"], scatter_kws={'alpha': 0.4})
    
    plt.title(f"{feature} SHAP vs Residual\nPearson r = {r:.3f}, p = {p:.4f}", fontsize=13)
    plt.xlabel(f"{feature} SHAP Value")
    plt.ylabel("Residual (Predicted - Actual KDE)")
    plt.grid(True)
    
    # 保存图像
    save_path = os.path.join(output_path, f"shap_vs_residual_{feature}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 打印结果（可选）
    print(f"{feature}: r = {r:.3f}, p = {p:.4f} → saved to {save_path}")