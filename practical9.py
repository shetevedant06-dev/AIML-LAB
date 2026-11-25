import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)

iris_dataset = load_iris()
features_df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
target_series = pd.Series(iris_dataset.target)

print("Iris dataset loaded successfully!")
print("Dataset shape:", features_df.shape)
print("\nFirst five records:")
print(features_df.head())
print("\nTarget classes:", iris_dataset.target_names)

x_train, x_test, y_train, y_test = train_test_split(
    features_df,
    target_series,
    test_size=0.20,
    random_state=42,
    stratify=target_series,
)

print("\nTraining samples:", x_train.shape[0])
print("Testing samples:", x_test.shape[0])

nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
print("\nGaussian Naive Bayes model trained successfully!")

pred_labels = nb_classifier.predict(x_test)

acc = accuracy_score(y_test, pred_labels)
precision_val = precision_score(y_test, pred_labels, average="weighted")
recall_val = recall_score(y_test, pred_labels, average="weighted")

print("\nModel Accuracy:", round(acc * 100, 2), "%")
print("Weighted Precision:", round(precision_val, 4))
print("Weighted Recall:", round(recall_val, 4))

conf_mat = confusion_matrix(y_test, pred_labels)
print("\nConfusion Matrix:")
print(conf_mat)

plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris_dataset.target_names,
    yticklabels=iris_dataset.target_names,
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_labels,
        target_names=iris_dataset.target_names,
    )
)