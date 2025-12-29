import random
import numpy as np
import tensorflow as tf
import torch
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import io



# Set a seed value
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator
tf.random.set_seed(seed_value)
# 5. Set `torch` pseudo-random generator
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value) # if you are using GPU



# Load parquet files
# Make sure to reference download instructions. You have to download the kaggle dataset and upload it, rename it to a data directory. Video guide coming soon
#from: https://drive.google.com/drive/folders/12-1XR8df-rYkwJuMqMgXYQ9dm6EoDx0f?usp=drive_link
#quickstart guide: https://www.youtube.com/watch?v=WPRarAeelAM
#Original source + description: https://advp.niagads.org/downloads
train_df = pd.read_parquet("models/mri_model/data/train.parquet")
test_df  = pd.read_parquet("models/mri_model/data/test.parquet")

def bytes_to_pixels(b: bytes) -> np.ndarray:
    """
    Convert raw image bytes (e.g. JPEG/PNG) into a 2D numpy array of pixel values (grayscale).
    """
    img = Image.open(io.BytesIO(b))  # convert to grayscale
    return np.array(img)
def extract_bytes(blob):
    """
    Unwrap a dict‐wrapped binary payload if needed,
    otherwise return blob directly.
    """
    if isinstance(blob, dict):
        # try common keys
        for key in ("bytes", "data", "image"):
            if key in blob and isinstance(blob[key], (bytes, bytearray)):
                return blob[key]
        # fallback: first bytes‐like value
        for v in blob.values():
            if isinstance(v, (bytes, bytearray)):
                return v
        raise TypeError(f"No bytes found in dict payload: {list(blob.keys())}")
    return blob

train_df["image"] = train_df["image"].apply(lambda blob: bytes_to_pixels(extract_bytes(blob)))
test_df["image"]  = test_df["image"].apply(lambda blob: bytes_to_pixels(extract_bytes(blob)))



# '0': Mild_Demented

# '1': Moderate_Demented

# '2': Non_Demented

# '3': Very_Mild_Demented


def add_gaussian_noise_batch(X, std=20):
    X_noisy = []
    for img in X:
        noise = np.random.normal(0, std/255.0, img.shape)
        noisy = np.clip(img + noise, 0, 1)
        X_noisy.append(noisy)
    return np.array(X_noisy)








# Build arrays from ORIGINAL images only
X = np.array([img.reshape(128,128,1) for img in train_df['image']])
y = tf.keras.utils.to_categorical(train_df['label'], num_classes=4)

X = X.astype('float32') / 255.0

# Split BEFORE augmentation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=train_df['label']
)



X_train_noisy = add_gaussian_noise_batch(X_train, std=20)

X_train = np.concatenate([X_train, X_train_noisy], axis=0)
y_train = np.concatenate([y_train, y_train], axis=0)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')  # 4 classes: 0–3
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, y_val)
)


model.save("models/mri_model/alzheimers_mri_model.keras")