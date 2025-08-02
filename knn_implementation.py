# KNN Model Implementation for Breast Cancer Prediction
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("[Dataset]_BreastCancer.csv")

# Data preprocessing
# Remove missing values
df = df.dropna()

# Rename diagnosis column to target
df = df.rename(columns={'diagnosis':'target'})

# Convert target to numeric (B=0, M=1)
df['target'] = df['target'].map({'B': 0, 'M': 1})

# Separate features and target
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model Implementation
print("=== KNN Model Implementation ===")

# Create and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = knn_model.predict(X_train_scaled)
y_pred_test = knn_model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Print results
print(f"KNN Model Results:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Hyperparameter tuning for better performance
print("\n=== Hyperparameter Tuning ===")

# Try different values of k
k_values = [3, 5, 7, 9, 11, 13, 15]
knn_scores = []

for k in k_values:
    knn_tuned = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    knn_tuned.fit(X_train_scaled, y_train)
    score = knn_tuned.score(X_test_scaled, y_test)
    knn_scores.append(score)
    print(f"k={k}: Test Accuracy = {score:.4f}")

# Find best k
best_k = k_values[np.argmax(knn_scores)]
best_score = max(knn_scores)
print(f"\nBest k value: {best_k} with accuracy: {best_score:.4f}")

# Plot k vs accuracy
plt.figure(figsize=(8, 6))
plt.plot(k_values, knn_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Test Accuracy')
plt.title('KNN: k vs Accuracy')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.show()

# Final model with best k
final_knn = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', metric='euclidean')
final_knn.fit(X_train_scaled, y_train)

final_train_accuracy = final_knn.score(X_train_scaled, y_train)
final_test_accuracy = final_knn.score(X_test_scaled, y_test)

print(f"\n=== Final KNN Model (k={best_k}) ===")
print(f"Training Accuracy: {final_train_accuracy:.4f}")
print(f"Test Accuracy: {final_test_accuracy:.4f}")

# Feature importance analysis
print("\n=== Feature Importance Analysis ===")
# For KNN, we can analyze which features contribute most to the classification
# by looking at the correlation with the target
correlation_with_target = X.corrwith(y).abs().sort_values(ascending=False)
print("Top 10 features by correlation with target:")
print(correlation_with_target.head(10))

# Plot feature correlations
plt.figure(figsize=(12, 8))
correlation_with_target.head(15).plot(kind='bar')
plt.title('Feature Correlation with Target')
plt.xlabel('Features')
plt.ylabel('Absolute Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 