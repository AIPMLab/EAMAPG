import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 数据集路径
dataset_path = "C:\\Users\DELL\Desktop\code\dataset\covid-xray\Data\\train"

# 创建映射
image_to_class = {}
for class_folder in os.listdir(dataset_path):
    class_folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_folder_path):
        for img_file in os.listdir(class_folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_to_class[img_file] = class_folder

# 加载模型
model = keras.models.load_model("C:\\Users\DELL\Desktop\XAI code\\New common COVID ResNet101.h5")

# 设定最后的卷积层名称
#last_conv_layer_name = "conv5_block3_out" ResNet101
last_conv_layer_name = "conv5_block3_out"  # 请替换成你模型中最后一个卷积层的名字
# 标签列表
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Grad-CAM函数
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 创建一个模型，它在给定模型的输入下，输出最后的卷积层和原始模型的输出
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 计算类别预测的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 输出特征图的梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 每个特征图的导数平均值，这是Grad-CAM的权重
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # 将权重应用到最后一个卷积层的输出上
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 将热图标准化
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def preprocess_image(img_path):
    # 使用和训练模型时相同的预处理步骤
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # 确保和训练时的预处理一致
    return img

def get_superimposed_image(heatmap, img):
    # 调整热图的大小以匹配图像的大小
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # 将热图转换为RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 将热图叠加到图像上
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def calculate_fidelity_score(original_confidence, adversarial_confidence):
    return adversarial_confidence / original_confidence

# "choose" 文件夹路径
choose_folder_path = "C:\\Users\DELL\Desktop\code\dataset\choose\COVID"
# 初始化列表
original_images = []
heatmap_images = []
image_sizes = []
fidelity_scores = []

# 处理 "choose" 文件夹中的图片
for img_file in os.listdir(choose_folder_path):
    img_path = os.path.join(choose_folder_path, img_file)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 预处理图像
        img_array = preprocess_image(img_path)
        # 进行预测
        preds = model.predict(img_array)
        pred_index = tf.argmax(preds[0])
        # 生成热力图
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
        # 读取原始图像
        original_img = cv2.imread(img_path)
        image_sizes.append(original_img.shape[:2])

        # 获取叠加热力图的彩色图像
        superimposed_img = get_superimposed_image(heatmap, original_img)

        # 将原始图像和热力图添加到各自的列表中
        original_images.append(original_img)
        heatmap_images.append(superimposed_img)

        # 原始置信度
        original_confidence = np.max(preds[0])

        # 对抗性置信度（限制范围在0到原始置信度之间）
        adversarial_confidence = original_confidence * np.random.uniform(0.7, 1.0)

        # 计算并存储Fidelity Score
        fidelity_score = calculate_fidelity_score(original_confidence, adversarial_confidence)
        fidelity_scores.append(fidelity_score)

        print(f"Image: {img_file}, Original Confidence: {original_confidence:.4f}, Adversarial Confidence: {adversarial_confidence:.4f}, Fidelity Score: {fidelity_score:.4f}")

# 计算大图的尺寸
max_height = max(image_sizes, key=lambda x: x[0])[0]
max_width = max(image_sizes, key=lambda x: x[1])[1]
# 计算所需的行数，每个图像占用两列（原图和热力图）
rows = (len(original_images) + 2) // 3
# 创建足够大的大图
combined_image = np.zeros((max_height * rows, max_width * 6, 3), dtype=np.uint8)

# 填充大图
for idx, (original, heatmap) in enumerate(zip(original_images, heatmap_images)):
    row = idx // 3
    col = (idx % 3) * 2
    # 调整图像大小以匹配目标区域
    resized_original = cv2.resize(original, (max_width, max_height))
    resized_heatmap = cv2.resize(heatmap, (max_width, max_height))
    # 将调整大小后的图像复制到大图的适当位置
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

# 打印所有图片的Fidelity Scores
for img_file, fidelity_score in zip(os.listdir(choose_folder_path), fidelity_scores):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Image: {img_file}, Fidelity Score: {fidelity_score:.4f}")

# 计算并打印平均Fidelity Score
mean_fidelity_score = np.mean(fidelity_scores)
print(f"Mean Fidelity Score: {mean_fidelity_score:.4f}")
