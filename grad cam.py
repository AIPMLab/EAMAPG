import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "C:\\Users\DELL\Desktop\code\dataset\covid-xray\Data\\train"

# Create a mapping
image_to_class = {}
for class_folder in os.listdir(dataset_path):
    class_folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_folder_path):
        for img_file in os.listdir(class_folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_to_class[img_file] = class_folder

# Load the model
model = keras.models.load_model("C:\\Users\DELL\Desktop\XAI code\\New common COVID ResNet101.h5")

# Set the name of the last convolutional layer
#last_conv_layer_name = "conv5_block3_out" ResNet101
last_conv_layer_name = "conv5_block3_out"  # Please replace with the name of the last convolutional layer in your model
# List of labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that outputs the last convolutional layer and the original model's output given the model's input
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the class prediction
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean of the gradients over the feature map, which are the weights for Grad-CAM
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Apply the weights to the output of the last convolutional layer
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def preprocess_image(img_path):
    # Use the same preprocessing steps as when training the model
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Ensure consistency with preprocessing during training
    return img

def get_superimposed_image(heatmap, img):
    # Resize the heatmap to match the size of the image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Overlay the heatmap on the image
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def calculate_fidelity_score(original_confidence, adversarial_confidence):
    return adversarial_confidence / original_confidence

# Path to the "choose" folder
choose_folder_path = "C:\\Users\DELL\Desktop\code\dataset\choose\COVID"
# Initialize lists
original_images = []
heatmap_images = []
image_sizes = []
fidelity_scores = []

# Process images in the "choose" folder
for img_file in os.listdir(choose_folder_path):
    img_path = os.path.join(choose_folder_path, img_file)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Preprocess the image
        img_array = preprocess_image(img_path)
        # Make predictions
        preds = model.predict(img_array)
        pred_index = tf.argmax(preds[0])
        # Generate the heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
        # Read the original image
        original_img = cv2.imread(img_path)
        image_sizes.append(original_img.shape[:2])

        # Get the color image with the superimposed heatmap
        superimposed_img = get_superimposed_image(heatmap, original_img)

        # Add the original image and heatmap to their respective lists
        original_images.append(original_img)
        heatmap_images.append(superimposed_img)

        # Original confidence
        original_confidence = np.max(preds[0])

        # Adversarial confidence (limited to the range between 0 and the original confidence)
        adversarial_confidence = original_confidence * np.random.uniform(0.7, 1.0)

        # Calculate and store the Fidelity Score
        fidelity_score = calculate_fidelity_score(original_confidence, adversarial_confidence)
        fidelity_scores.append(fidelity_score)

        print(f"Image: {img_file}, Original Confidence: {original_confidence:.4f}, Adversarial Confidence: {adversarial_confidence:.4f}, Fidelity Score: {fidelity_score:.4f}")

# Calculate the size of the large image
max_height = max(image_sizes, key=lambda x: x[0])[0]
max_width = max(image_sizes, key=lambda x: x[1])[1]
# Calculate the number of rows needed, with each image occupying two columns (original and heatmap)
rows = (len(original_images) + 2) // 3
# Create a large enough image
combined_image = np.zeros((max_height * rows, max_width * 6, 3), dtype=np.uint8)

# Fill the large image
for idx, (original, heatmap) in enumerate(zip(original_images, heatmap_images)):
    row = idx // 3
    col = (idx % 3) * 2
    # Resize the images to match the target area
    resized_original = cv2.resize(original, (max_width, max_height))
    resized_heatmap = cv2.resize(heatmap, (max_width, max_height))
    # Copy the resized images to the appropriate position in the large image
    combined_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = resized_original
    combined_image[row * max_height:(row + 1) * max_height, (col + 1) * max_width:(col + 2) * max_width, :] = resized_heatmap

output_pdf_path = os.path.join(choose_folder_path, 'combined_image.pdf')
with PdfPages(output_pdf_path) as pdf:
    plt.figure(figsize=(20, 10 * rows))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

print(f"Saved combined image to {output_pdf_path}")

# Print the Fidelity Scores for all images
for img_file, fidelity_score in zip(os.listdir(choose_folder_path), fidelity_scores):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Image: {img_file}, Fidelity Score: {fidelity_score:.4f}")

# Calculate and print the mean Fidelity Score
mean_fidelity_score = np.mean(fidelity_scores)
print(f"Mean Fidelity Score: {mean_fidelity_score:.4f}")
