import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
MODEL_PATH = "saved_model/brain_tumor_resnet50_final.keras"  # Change path if necessary
model = tf.keras.models.load_model(MODEL_PATH)

# Define test data directory
TEST_DIR = "dataset/test"  # Ensure correct path
IMAGE_SIZE = (224, 224)  # Ensure correct input size

# Create test image generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,  # Fixed input size to 224x224
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# Function to predict on a single image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)  # Fixed to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)
    class_labels = list(test_generator.class_indices.keys())  # Get class labels
    predicted_class = class_labels[np.argmax(prediction)]  # Get predicted class
    confidence = np.max(prediction) * 100  # Confidence score

    print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    return predicted_class, confidence

# Test single image prediction
TEST_IMAGE_PATH = "dataset/test/glioma/Te-gl_0010.jpg"  # Change to an actual image path
if os.path.exists(TEST_IMAGE_PATH):
    predict_image(TEST_IMAGE_PATH)
else:
    print("Test image not found. Please update TEST_IMAGE_PATH.")
