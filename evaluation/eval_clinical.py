import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import json
import os

# Reload dataset
final_tau = pd.read_csv("models/clinical_model/data/final_tau.csv")

X = final_tau.iloc[:,:-1]
y = final_tau.iloc[:,-1]

# Recreate same split
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)

# Load model
bst = xgb.Booster()
bst.load_model("models/clinical_model/data/alzheimers_clinical_model.json")

dtest = xgb.DMatrix(X_test)
probs = bst.predict(dtest)  # shape: (n_samples, 3)
y_pred = probs.argmax(axis=1)

# Metrics
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

print("Clinical Model Metrics")
print("Weighted F1:", f1)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["CN","MCI","AD"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save metrics
os.makedirs("results", exist_ok=True)

with open("results/clinical_metrics.json","w") as f:
    json.dump({
        "weighted_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist()
    }, f, indent=2)

# Save confusion matrix as image
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=["CN","MCI","AD"],
            yticklabels=["CN","MCI","AD"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Clinical Model Confusion Matrix")
plt.savefig("results/clinical_confusion_matrix.png")
plt.close()