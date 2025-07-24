import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# ==== 1. Load dataset and create DataFrame ====
dataset_dir =  r'C:\Users\Tippu\Desktop\Roshan\Pattern\data_pattern'
image_paths = []
labels = []

for class_label in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_label)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_path, filename))
                labels.append(class_label)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==== 2. Split into train/validation/test ====
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Total: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

# ==== 3. Preprocess images manually ====
IMG_SIZE = (180, 180)

def load_images(df):
    images = []
    labels = []
    for _, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # Normalize
        images.append(img)
        labels.append(row['label'])
    return np.array(images), np.array(labels)

X_train, y_train = load_images(train_df)
X_valid, y_valid = load_images(valid_df)
X_test, y_test = load_images(test_df)

# ==== 4. One-hot encode labels ====
lb = LabelBinarizer()
y_train_encoded = lb.fit_transform(y_train)
y_valid_encoded = lb.transform(y_valid)
y_test_encoded = lb.transform(y_test)

print("Image shapes:", X_train.shape, X_valid.shape, X_test.shape)
print("Label shapes:", y_train_encoded.shape, y_valid_encoded.shape, y_test_encoded.shape)

# ==== 5. Optional: Show class distribution ====
sns.countplot(x='label', data=train_df)
plt.title("Class Distribution in Train Set")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==== ✅ CONNECT: Handing off preprocessed data to model ====
print("\n✅ Data preprocessing complete. Proceeding to model building...")

# Confirm data shapes before model input
print("X_train shape:", X_train.shape)
print("y_train_encoded shape:", y_train_encoded.shape)
print("Number of classes:", y_train_encoded.shape[1])

# Ensure NumPy arrays are of correct types for PyTorch
assert X_train.ndim == 4 and X_train.shape[1:] == (180, 180, 3), "Images must be of shape (180, 180, 3)"
assert y_train_encoded.ndim == 2, "Labels must be one-hot encoded"

# Set the random seed for reproducibility
import random
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📦 Using device:", device)
