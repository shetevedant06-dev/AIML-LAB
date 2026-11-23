import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("========== MATH LIBRARY ==========")
num = 36
print("Square root of", num, ":", math.sqrt(num))
print("Power (2^5):", math.pow(2, 5))
print("Factorial of 5:", math.factorial(5))
print("Value of Pi:", round(math.pi, 4))
print("Ceiling of 4.3:", math.ceil(4.3))
print("Floor of 4.7:", math.floor(4.7))

print("\n========== NUMPY LIBRARY ==========")
scores = np.array([68, 74, 80, 76, 88])
print("Scores:", scores)
print("Sum:", np.sum(scores))
print("Mean:", np.mean(scores))
print("Median:", np.median(scores))
print("Standard Deviation:", round(float(np.std(scores)), 2))
print("Maximum:", np.max(scores))
print("Minimum:", np.min(scores))

print("\n========== SCIPY LIBRARY ==========")
z_scores = stats.zscore(scores)
print("Z-Scores:", np.round(z_scores, 2))
mode_res = stats.mode(scores, keepdims=True)
print("Mode:", mode_res.mode[0])

print("\n========== PANDAS LIBRARY ==========")
records = {
    "Student": ["Amit", "Neha", "Rahul", "Sneha", "Priya"],
    "Maths": [78, 85, 72, 90, 88],
    "Python": [82, 90, 75, 95, 86],
    "AI": [80, 88, 70, 92, 90],
}
df = pd.DataFrame(records)
print("\nStudent Records:")
print(df)

df["Average"] = df[["Maths", "Python", "AI"]].mean(axis=1).round(1)
print("\nRecords with Average:")
print(df)

print("\nDescriptive Statistics:")
print(df.describe())

print("\n========== MATPLOTLIB VISUALIZATION ==========")
plt.figure(figsize=(8, 5))
plt.bar(df["Student"], df["Average"], color="#4A90E2")
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Average Marks of Students")
plt.tight_layout()
plt.show()

print("\n========== SEABORN VISUALIZATION ==========")
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Maths", y="Python", data=df, s=100)
plt.title("Maths vs Python Marks")
plt.xlabel("Maths Marks")
plt.ylabel("Python Marks")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
corr_matrix = df[["Maths", "Python", "AI"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Subject Correlation Heatmap")
plt.tight_layout()
plt.show()