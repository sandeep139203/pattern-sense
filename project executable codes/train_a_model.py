import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# ==== 1. Dataset path ====
DATASET_DIR = r'C:\Users\Tippu\Desktop\Roshan\Pattern\data_pattern'  # <-- Replace if your folder is named differently
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 7

# ==== 2. Load and preprocess images ====
def load_dataset(dataset_dir):
    image_paths, labels = [], []
    for class_label in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, filename))
                    labels.append(class_label)
    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_images(df):
    images, labels = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
        labels.append(row['label'])
    return np.array(images), np.array(labels)

df = load_dataset(DATASET_DIR)

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

X_train, y_train = load_images(train_df)
X_valid, y_valid = load_images(valid_df)
X_test, y_test = load_images(test_df)

# One-hot encode labels
lb = LabelBinarizer()
y_train_encoded = lb.fit_transform(y_train)
y_valid_encoded = lb.transform(y_valid)
y_test_encoded = lb.transform(y_test)
class_names = lb.classes_

# ==== 3. Custom PyTorch Dataset ====
class FabricDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.transpose(0, 3, 1, 2).astype(np.float32)
        self.labels = np.argmax(labels, axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx]).long()

train_data = FabricDataset(X_train, y_train_encoded)
valid_data = FabricDataset(X_valid, y_valid_encoded)
test_data = FabricDataset(X_test, y_test_encoded)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# ==== 4. Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(class_names))
)
model = model.to(device)

# ==== 5. Train Model ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ==== 6. Save Model and Class Labels ====
torch.save(model.state_dict(), "fabric_pattern_model.pt")
np.save("classes.npy", class_names)
print("✅ Model saved as fabric_pattern_model.pt")
print("✅ Class names saved as classes.npy")

# ==== 7. Evaluate on Test Set ====
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

acc = accuracy_score(y_true, y_pred)
print(f"🎯 Test Accuracy: {acc * 100:.2f}%")
