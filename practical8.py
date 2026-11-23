import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

dataset = load_breast_cancer()
features_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
target_series = pd.Series(dataset.target)

print("Dataset loaded successfully!")
print("Dataset shape:", features_df.shape)
print("\nFirst five records:")
print(features_df.head())
print("\nTarget classes:", dataset.target_names)
print("0 = Malignant, 1 = Benign")

x_train, x_test, y_train, y_test = train_test_split(
    features_df,
    target_series,
    test_size=0.20,
    random_state=42,
    stratify=target_series,
)

print("\nTraining samples:", x_train.shape[0])
print("Testing samples:", x_test.shape[0])

feature_scaler = StandardScaler()
x_train_scaled = feature_scaler.fit_transform(x_train)
x_test_scaled = feature_scaler.transform(x_test)

classifier = LogisticRegression(max_iter=5000)
classifier.fit(x_train_scaled, y_train)
print("\nLogistic Regression model trained successfully!")

pred_labels = classifier.predict(x_test_scaled)
pred_probs = classifier.predict_proba(x_test_scaled)[:, 1]

acc = accuracy_score(y_test, pred_labels)
print("\nModel Accuracy:", round(acc * 100, 2), "%")

conf_mat = confusion_matrix(y_test, pred_labels)
print("\nConfusion Matrix:")
print(conf_mat)

plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=dataset.target_names,
    yticklabels=dataset.target_names,
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_labels,
        target_names=dataset.target_names,
    )
)

fpr, tpr, thresholds = roc_curve(y_test, pred_probs)
auc_val = roc_auc_score(y_test, pred_probs)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("ROC-AUC Score:", round(auc_val, 4))