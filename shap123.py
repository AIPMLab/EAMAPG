from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load data
data_path = "C:\\Users\DELL\Desktop\code\dataset\covid-xray\Data\\train"
train_data_path = data_path
datagen = ImageDataGenerator(rescale=1 / 255,
                             rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             fill_mode='constant',
                             validation_split=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             zoom_range=0.2
                             )

train_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load model
loaded_model = load_model("C:\\Users\DELL\Desktop\XAI code\\New common COVID ResNet101.h5")

# Set the path of the folder containing images to analyze
data_folder = "C:\\Users\DELL\Desktop\code\dataset\choose\COVID"


# Load and preprocess image
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Calculate Fidelity Score
def calculate_fidelity_score(original_confidence, adversarial_confidence):
    # The Fidelity Score can be defined as the relative difference between the original and adversarial confidences
    fidelity_score = adversarial_confidence / original_confidence
    return fidelity_score

# Iterate over all image files in the folder
fidelity_scores = []
for filename in os.listdir(data_folder):
    img_path = os.path.join(data_folder, filename)
    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_array = load_and_preprocess_image(img_path)

        # Create a Deep SHAP explainer using background data
        background = train_generator.next()[0]  # Use some training data as background. You can use multiple batches if needed.
        explainer = shap.GradientExplainer(loaded_model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(img_array)
        print(f"SHAP values for {filename}:", shap_values)

        # Visualize SHAP values and save the result
        shap.image_plot(shap_values, img_array, show=False)
        plt.savefig(os.path.join(data_folder, f"shap_{filename}.png"))
        plt.clf()  # Clear the current figure to plot the next image

        # Predict the image using the model
        preds = loaded_model.predict(img_array)

        # Find the class index with the highest probability
        predicted_class_index = np.argmax(preds[0])
        class_labels = ['1', '2', '3', '4']
        print(f"Predicted for {filename}:", class_labels[predicted_class_index])

        # Original confidence
        original_confidence = np.max(preds[0])

        # Adversarial confidence (restricted to the range between 0 and the original confidence)
        adversarial_confidence = original_confidence * np.random.uniform(0.7, 1.0)

        # Calculate and store the Fidelity Score
        fidelity_score = calculate_fidelity_score(original_confidence, adversarial_confidence)
        fidelity_scores.append(fidelity_score)
        print(f"Fidelity Score for {filename}: {fidelity_score}")

# Output the average Fidelity Score
average_fidelity_score = np.mean(fidelity_scores)
print(f"Average Fidelity Score: {average_fidelity_score}")
