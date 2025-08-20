import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("Data_cleared/final_dataset_imputed.csv")  # Replace with your actual file
output_dir = "Output/1. EDA"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Fig 1: Distribution Summary
# -----------------------------
features = ['crime_count', 'ptal_score', 'imd_score']
plt.figure(figsize=(12, 4))
for i, var in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[var], kde=True, bins=30, color='steelblue')
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{output_dir}/distribution_summary.png")
plt.close()

# -----------------------------
# Fig 2: Boxplots by Hotspot_10
# -----------------------------
selected_vars = ['ptal_score', 'imd_score', 'pop_density']
df['hotspot_10'] = df['hotspot_10'].astype(int)

plt.figure(figsize=(12, 4))
for i, var in enumerate(selected_vars, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df, x='hotspot_10', y=var, palette='Set2')
    plt.title(f'{var} by Hotspot_10')
    plt.xlabel('Hotspot_10 (0=No, 1=Yes)')
    plt.ylabel(var)
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_by_hotspot10.png")
plt.close()

# -----------------------------
# Fig 3: Correlation Heatmap
# -----------------------------
corr_vars = ['crime_count', 'ptal_score', 'imd_score', 'pop_density', 'road_density', 'node_density']
corr = df[corr_vars].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Pearson Correlation Among Features")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_correlation_heatmap.png")
plt.close()

print("EDA figures (3) saved to './eda_output' folder.")