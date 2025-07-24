from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
class_names = np.load("classes.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(class_names))
)
model.load_state_dict(torch.load("fabric_pattern_model.pt", map_location=device))
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor()
])
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/app")
def app_page():
    return render_template("app.html", prediction=None, image_path=None)
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file and file.filename != "":
        filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            prediction = class_names[predicted.item()]

        return render_template("app.html", prediction=prediction, image_path=filepath)

    return render_template("app.html", prediction="No file selected", image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
