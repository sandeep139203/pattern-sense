import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set path to your dataset folder
dataset_dir = r'C:\Users\Tippu\Desktop\Roshan\Pattern\data_pattern'

# List to hold image paths and labels
image_paths = []
labels = []

# Walk through each subfolder (class)
for class_label in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_label)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_path, filename))
                labels.append(class_label)

# Create DataFrame
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train (70%), validation (15%), and test (15%)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Output
print(f"Total images: {len(df)}")
print(f"Train: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}")

# Optional preview
print(train_df.head())
