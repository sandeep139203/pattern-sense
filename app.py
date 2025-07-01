from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load trained fabric pattern model
model = load_model("psm_best_model.h5")  # Updated model name

# Fabric pattern classes
classes = [
    'Animal', 'Cartoon', 'Floral', 'Geometry', 'Ikat',
    'Plain', 'Polka Dot', 'Squares', 'Stripes', 'Tribal'
]

# Set up upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction function
def predict_pattern(img_path):
    try:
        print(f"🔍 Processing: {img_path}")

        # Load and preprocess image to match model input (224x224)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        prediction = model.predict(img_array, verbose=0)[0]
        top_class = np.argmax(prediction)
        confidence = prediction[top_class]

        # Debug output
        print("✅ Prediction vector:")
        for i, prob in enumerate(prediction):
            print(f"{classes[i]:12s}: {prob:.4f} ({prob*100:.2f}%)")
        print(f"🎯 Final prediction: {classes[top_class]} ({confidence*100:.2f}%)")

        return f"{classes[top_class]} ({confidence*100:.1f}% confidence)"
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error: {str(e)}"

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/industries')
def industries():
    return render_template('industries.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == "":
            return render_template('predict.html', prediction="⚠️ No valid file uploaded.")

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        print("\n" + "="*50)
        print(f"🖼️ NEW PREDICTION REQUEST — {filename}")
        print("="*50)

        prediction = predict_pattern(path)
        img_path = '/' + path.replace('\\', '/')

        return render_template('predict.html', prediction=prediction, img_path=img_path)

    return render_template('predict.html')

# Run the app
if __name__ == '__main__':
    print("✅ Pattern Sense running at: http://127.0.0.1:5000")
    print("🚀 Debug mode active")
    app.run(debug=True)
