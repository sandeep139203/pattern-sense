# 🧵 Pattern Sense: Fabric Pattern Classifier

A web-based deep learning app to classify fabric patterns using PyTorch and Flask.

---

## 📁 Project Structure
```
Pattern/
│
├── app.py                    # Flask web application
├── train_model.py           # Model training script
├── generate_dummy_model.py  # Creates a dummy model file
├── data_pre_processing_models.py  # Data preprocessing and analysis
├── create_data_labels.py    # CSV-style label creation
├── fabric_pattern_model.pt  # Trained PyTorch model (output)
├── classes.npy              # Saved class labels (output)
├── templates/
│   ├── home.html            # Homepage HTML
│   └── app.html             # Upload and result page
└── data_pattern/            # Dataset directory (with subfolders for each class)
```

---

## ✅ Requirements

Install the required Python packages:

```bash
pip install torch torchvision flask numpy pandas scikit-learn opencv-python matplotlib seaborn
```

---

## 📸 Dataset Structure

Place your images in the following structure:

```
data_pattern/
├── checked/
│   ├── img1.jpg
│   └── ...
├── floral/
├── striped/
├── zigzag/
└── ...
```

Each subfolder is treated as a class.

---

## 🔧 Steps to Run

### 1. **Prepare and preprocess data**
```bash
python data_pre_processing_models.py
```

### 2. **Train the model**
```bash
python train_model.py
```

> This saves:
> - `fabric_pattern_model.pt`
> - `classes.npy`

### 3. **Run the Flask app**
```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)  
Upload an image to classify its fabric pattern.

---

## 🛠 Optional

- To generate a random initialized model (for testing only):
```bash
python generate_dummy_model.py
```

- To view label distribution or split logic:
```bash
python create_data_labels.py
```

---

## 🚀 Features

- Upload fabric image and get class prediction
- CNN-based classifier using ResNet18
- Training data split into Train/Validation/Test
- Clean and simple web UI with Flask

---

## 📌 Notes

- Supports `.jpg`, `.png`, `.jpeg` images
- Images are resized to **180×180**
- Doesn't save uploaded images on the server

---

## 🧠 Example Classes

```txt
['checked', 'floral', 'striped', 'zigzag', 'geometric', 'abstract', ...]
```

You can customize these in your dataset directory.

---

## 👨‍💻 Author

**Shaik Roshan**  
Project: *Pattern Sense - Fabric Pattern Classifier*"# Fabric-Pattern-Sense" 
