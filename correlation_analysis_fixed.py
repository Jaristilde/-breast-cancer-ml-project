# Fixed Correlation Analysis for Breast Cancer Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("[Dataset]_BreastCancer.csv")

# Data preprocessing
# Remove missing values
df = df.dropna()

# Rename diagnosis column to target
df = df.rename(columns={'diagnosis':'target'})

# Convert target to numeric (B=0, M=1)
df['target'] = df['target'].map({'B': 0, 'M': 1})

# Calculate correlation matrix
corr = df.corr()

# Analyze correlations with different thresholds
for th in [0.6, 0.75, 0.85]:
    filt = np.abs(corr['target']) > th
    top_features = corr.columns[filt]
    print(f"Features with correlation > {th}:")
    print(top_features.tolist())
    print("-" * 40)

# Focus on the most relevant threshold
th = 0.75
filt = np.abs(corr['target']) > th
top_corr = df.loc[:, filt]

plt.figure(figsize=(10, 8))
sns.heatmap(top_corr.corr(), annot=True, cmap='YlGnBu')
plt.title(f"Top Features with Correlation > {th}")
plt.show()

# Additional analysis: Show correlation values with target
print("\nCorrelation values with target:")
target_correlations = corr['target'].abs().sort_values(ascending=False)
print(target_correlations.head(10)) 