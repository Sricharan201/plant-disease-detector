import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

MODEL_PATH = "model/plant_disease_model.keras"

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

model = tf.keras.models.load_model(MODEL_PATH)

if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("Example: python test_model.py test_leaf.jpg")
    sys.exit()

image_path = sys.argv[1]

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index] * 100

print()
print("Prediction:", class_names[predicted_index])
print("Confidence:", round(confidence, 2), "%")
print()

print("Top predictions:")

top_indices = np.argsort(predictions[0])[::-1][:3]

for index in top_indices:
    print(
        f"{class_names[index]}: "
        f"{predictions[0][index] * 100:.2f}%"
    )