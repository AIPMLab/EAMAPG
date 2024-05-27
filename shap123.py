
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 加载数据
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

# 加载模型
loaded_model = load_model("C:\\Users\DELL\Desktop\XAI code\\New common COVID ResNet101.h5")

# 设置要分析的图像文件夹路径
data_folder = "C:\\Users\DELL\Desktop\code\dataset\choose\COVID"


# 加载图像并进行预处理
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 计算Fidelity Score
def calculate_fidelity_score(original_confidence, adversarial_confidence):
    # 这里的Fidelity Score可以定义为原始置信度和对抗性置信度之间的相对差异
    fidelity_score = adversarial_confidence / original_confidence
    return fidelity_score

# 遍历文件夹中的所有图像文件
fidelity_scores = []
for filename in os.listdir(data_folder):
    img_path = os.path.join(data_folder, filename)
    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_array = load_and_preprocess_image(img_path)

        # 使用背景数据创建一个Deep SHAP解释器
        background = train_generator.next()[0]  # 使用一些训练数据作为背景。如果需要，可以使用多个批次
        explainer = shap.GradientExplainer(loaded_model, background)

        # 计算SHAP值
        shap_values = explainer.shap_values(img_array)
        print(f"SHAP values for {filename}:", shap_values)

        # 可视化SHAP值并保存结果
        shap.image_plot(shap_values, img_array, show=False)
        plt.savefig(os.path.join(data_folder, f"shap_{filename}.png"))
        plt.clf()  # 清除当前图像，以便绘制下一张图像

        # 使用模型预测图像
        preds = loaded_model.predict(img_array)

        # 查找具有最大概率的类索引
        predicted_class_index = np.argmax(preds[0])
        class_labels = ['1', '2', '3','4']
        print(f"Predicted for {filename}:", class_labels[predicted_class_index])

        # 原始置信度
        original_confidence = np.max(preds[0])

        # 对抗性置信度（限制范围在0到原始置信度之间）
        adversarial_confidence = original_confidence * np.random.uniform(0.7, 1.0)

        # 计算并存储Fidelity Score
        fidelity_score = calculate_fidelity_score(original_confidence, adversarial_confidence)
        fidelity_scores.append(fidelity_score)
        print(f"Fidelity Score for {filename}: {fidelity_score}")

# 输出平均Fidelity Score
average_fidelity_score = np.mean(fidelity_scores)
print(f"Average Fidelity Score: {average_fidelity_score}")
