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

cancer_dataset = load_breast_cancer()
x_df = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)
y_series = pd.Series(cancer_dataset.target)

print("Dataset loaded successfully!")
print("Dataset shape:", x_df.shape)
print("\nFirst five records:")
print(x_df.head())
print("\nTarget classes:", cancer_dataset.target_names)
print("0 = Malignant, 1 = Benign")

x_train, x_test, y_train, y_test = train_test_split(
    x_df,
    y_series,
    test_size=0.20,
    random_state=42,
    stratify=y_series,
)

print("\nTraining samples:", x_train.shape[0])
print("Testing samples:", x_test.shape[0])

scaler_model = StandardScaler()
x_train_scaled = scaler_model.fit_transform(x_train)
x_test_scaled = scaler_model.transform(x_test)

clf_model = LogisticRegression(max_iter=5000)
clf_model.fit(x_train_scaled, y_train)
print("\nLogistic Regression model trained successfully!")

y_pred = clf_model.predict(x_test_scaled)
y_prob = clf_model.predict_proba(x_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(acc * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cancer_dataset.target_names,
    yticklabels=cancer_dataset.target_names,
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
        y_pred,
        target_names=cancer_dataset.target_names,
    )
)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Breast Cancer Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("ROC-AUC Score:", round(auc_score, 4))