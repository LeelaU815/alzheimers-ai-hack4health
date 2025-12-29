import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

model = tf.keras.models.load_model("models/mri_model/alzheimers_mri_model.keras")

test_df = pd.read_parquet("models/mri_model/data/test.parquet")

def extract_bytes(blob):
    # Unwrap dict-wrapped binary payload if needed
    if isinstance(blob, dict):
        for key in ("bytes", "data", "image"):
            if key in blob and isinstance(blob[key], (bytes, bytearray)):
                return blob[key]
        for v in blob.values():
            if isinstance(v, (bytes, bytearray)):
                return v
        raise TypeError(f"No bytes found in dict payload: {list(blob.keys())}")
    return blob

def bytes_to_pixels(blob):
    b = extract_bytes(blob)
    return np.array(Image.open(io.BytesIO(b)))

test_df['image'] = test_df['image'].apply(bytes_to_pixels)

# Build TEST set (never seen during training)
X_test = np.array([img.reshape(128,128,1) for img in test_df['image']]).astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(test_df['label'], num_classes=4)

loss, acc = model.evaluate(X_test, y_test)
print("True test accuracy:", acc)

# Get predictions
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
weighted_precision = precision_score(y_true, y_pred, average="weighted")
weighted_recall = recall_score(y_true, y_pred, average="weighted")  # sensitivity

print("\nMRI Classification Metrics")
print("-------------------------")
print("Weighted F1:", weighted_f1)
print("Weighted Precision:", weighted_precision)
print("Weighted Recall (Sensitivity):", weighted_recall)

print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=[
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# Saving metrics to a file
import json
import os

os.makedirs("results",exist_ok=True)

results = {
    "accuracy": float(acc),
    "weighted_f1": float(weighted_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
}

with open("results/mri_evaluation_metrics.json","w") as f:
    json.dump(results,f,indent=2)

# Saving confusion matrix as an image
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,fmt='d',
            xticklabels=[
                "Mild",
                "Moderate",
                "Non",
                "Very Mild"
            ],
            yticklabels=[
                "Mild",
                "Moderate",
                "Non",
                "Very Mild"
            ])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MRI Classification Confusion Matrix")

plt.savefig("results/mri_confusion_matrix.png")
plt.close()