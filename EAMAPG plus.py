import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

# 加载模型
model_path = "C:\\Users\DELL\Desktop\XAI code\eyes Densenet121.h5"
model = load_model(model_path)

# 类别标签
class_labels = ['1', '2', '3',"4"]

# 对抗生成攻击函数
def pgd_attack(model, input_image, target_label, epsilon=0.1, epsilon_step=0.01, num_steps=10):
    perturbed_image = input_image
    target_label = int(target_label)
    target_label_one_hot = tf.one_hot(target_label, depth=len(class_labels))
    target_label_one_hot = tf.reshape(target_label_one_hot, (1, -1))

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_image)
            prediction = model(perturbed_image, training=False)
            loss = tf.keras.losses.categorical_crossentropy(target_label_one_hot, prediction)

        gradient = tape.gradient(loss, perturbed_image)
        perturbation = epsilon_step * tf.sign(gradient)
        perturbed_image = perturbed_image + perturbation
        perturbed_image = tf.clip_by_value(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

    return perturbed_image

# 反向预处理函数
def reverse_preprocess_input(img_array):
    img_array += 1
    img_array *= 127.5
    return np.clip(img_array, 0, 255).astype('uint8')

# 生成对抗性样本函数
def generate_adversarial_example(model, input_image_path, target_label, epsilon=0.1, epsilon_step=0.01, num_steps=10):
    img = image.load_img(input_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    input_image = tf.convert_to_tensor(img_array, dtype=tf.float32)
    adversarial_image = pgd_attack(model, input_image, target_label, epsilon, epsilon_step, num_steps)
    return img_array, adversarial_image

# 处理预测结果函数
def process_predictions(predictions, class_labels):
    predicted_class_indices = np.argmax(predictions, axis=-1)
    predicted_labels = [class_labels[i] for i in predicted_class_indices]
    predicted_probabilities = np.max(predictions, axis=-1)
    return predicted_labels, predicted_probabilities

# 比较原始图像和对抗性图像函数
def compare_original_and_adversarial(model, original_image, adversarial_image):
    original_image_rev = reverse_preprocess_input(original_image)
    adversarial_image_rev = reverse_preprocess_input(adversarial_image.numpy())

    original_pred = model.predict(original_image)
    adversarial_pred = model.predict(adversarial_image)

    original_labels, original_confidence = process_predictions(original_pred, class_labels)
    adversarial_labels, adversarial_confidence = process_predictions(adversarial_pred, class_labels)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(original_image_rev[0].astype('uint8'))
    axs[0].set_title(f'Original: {original_labels[0]}\nConfidence: {original_confidence[0]:.2f}')
    axs[0].axis('off')

    axs[1].imshow(adversarial_image_rev[0].astype('uint8'))
    axs[1].set_title(f'Adversarial: {adversarial_labels[0]}\nConfidence: {adversarial_confidence[0]:.2f}')
    axs[1].axis('off')

    plt.show()

    return original_confidence[0], adversarial_confidence[0]

# 获取文件夹中的所有图片路径
def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
    return image_paths

folder_path = "C:\\Users\DELL\Desktop\code\dataset\choose\eyes"# 更新为你的实际文件夹路径
input_image_paths = get_image_paths(folder_path)

# 初始化列表以存储置信度
original_confidences = []
adversarial_confidences = []

# 生成对抗性样本并比较
for image_path in input_image_paths:
    original_image, adversarial_image = generate_adversarial_example(model, image_path, target_label=3, epsilon=0.1, epsilon_step=0.01, num_steps=10)
    original_confidence, adversarial_confidence = compare_original_and_adversarial(model, original_image, adversarial_image)
    original_confidences.append(original_confidence)
    adversarial_confidences.append(adversarial_confidence)
    print(f"Image: {image_path}, Original Confidence: {original_confidence}, Adversarial Confidence: {adversarial_confidence}")

# 检查是否有 NaN 或 Inf 值
def check_data(data):
    if np.any(np.isnan(data)):
        print("Data contains NaN values.")
    if np.any(np.isinf(data)):
        print("Data contains Inf values.")
    if np.all(data == data[0]):
        print("Data contains all identical values.")

print("Checking original_confidences:")
check_data(original_confidences)
print("Checking adversarial_confidences:")
check_data(adversarial_confidences)

# 进行 t-test
if len(original_confidences) > 1 and len(adversarial_confidences) > 1:
    t_stat, p_value = ttest_ind(original_confidences, adversarial_confidences)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print("There is a significant difference between the original and adversarial accuracies.")
    else:
        print("There is no significant difference between the original and adversarial accuracies.")
else:
    print("Not enough data to perform t-test.")

# 进行 ANOVA
if len(original_confidences) > 1 and len(adversarial_confidences) > 1:
    f_stat, p_value_anova = f_oneway(original_confidences, adversarial_confidences)
    print(f"ANOVA f-statistic: {f_stat}, p-value: {p_value_anova}")
    if p_value_anova < 0.05:
        print("There is a significant difference between the original and adversarial accuracies according to ANOVA.")
    else:
        print("There is no significant difference between the original and adversarial accuracies according to ANOVA.")
else:
    print("Not enough data to perform ANOVA.")

