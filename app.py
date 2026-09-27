from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("model/plant_disease_model.keras")

# IMPORTANT:
# This order must match the alphabetical folder order
# used by flow_from_directory() during training.
class_names = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename:

            # Save uploaded image
            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)

            upload_path = os.path.join(
                upload_folder,
                file.filename
            )

            file.save(upload_path)

            # Load and resize image
            img = image.load_img(
                upload_path,
                target_size=(224, 224)
            )

            # Convert image to array
            img_array = image.img_to_array(img)

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Normalize pixel values
            img_array = img_array / 255.0

            # Make prediction
            predictions = model.predict(
                img_array,
                verbose=0
            )

            # Get predicted class
            predicted_class = np.argmax(predictions[0])

            # Get confidence
            confidence = np.max(predictions[0]) * 100

            # Confidence threshold
            if confidence < 70:
                prediction = "Unable to confidently identify the disease"
            else:
                prediction = class_names[predicted_class]

            # Convert confidence to 2 decimal places
            confidence = round(confidence, 2)

            # Send image path to HTML
            image_path = upload_path

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_path=image_path
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)