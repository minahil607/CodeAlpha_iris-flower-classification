"""Task 1: Iris Flower Classification (using Iris.csv)
Put this file in the SAME folder as Iris.csv, then run it.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# STEP 1: Load the dataset
df = pd.read_csv("Iris.csv")
sns.pairplot(df.drop(columns="Id"), hue="Species")
plt.show()
print("Shape:", df.shape)
print(df.head(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
print("Class counts:\n", df["Species"].value_counts(), "\n")

# STEP 2: Choose features (X) and label (y). 'Id' is just a row number, so drop it.
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# STEP 3: Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# STEP 4: Train and compare models
models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:20s} CV accuracy: {cv:.3f} | Test accuracy: {acc:.3f}")

# STEP 5: Evaluate the chosen model in detail
best = models["KNN"]
pred = best.predict(X_test)
print("\nClassification report:\n", classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues", xticks_rotation=20)
plt.title("Iris KNN - Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()

# STEP 6: Predict a brand-new flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("\nPredicted species:", best.predict(new_flower)[0])
