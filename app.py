from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "saved_model/brain_tumor_resnet50_best.keras"  # Adjust the path if needed
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 expects (224,224)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    return CLASS_NAMES[max_index], confidence

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            prediction, confidence = predict_tumor(file_path)
            return render_template(
                "index.html",
                image=file.filename,
                result=prediction,
                tumor_status=f"Tumor Type: {prediction}",
                confidence=f"Confidence: {confidence}%",
            )

    return render_template("index.html", image=None, result=None)

if __name__ == "__main__":
    app.run(debug=True)
