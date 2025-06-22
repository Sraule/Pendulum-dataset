import numpy as np
from PIL import Image, ImageDraw
import os
import csv

IMG_SIZE = 224
NUM_SAMPLES = 7000
LIGHT_RADIUS = 40
PENDULUM_LENGTH = 30
OUTPUT_DIR = "./dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_pendulum_image(phi1, phi2):
    """
    输入：
    - phi1: 光源角度（弧度制），控制光源在顶部的位置
    - phi2: 摆锤角度（弧度制），控制摆的摆动方向

    输出：
    - image: RGB图像（PIL）
    - labels: 包含4个值的向量 [light_x, pendulum_angle, shadow_pos, shadow_len]
    """
    center = (IMG_SIZE // 2, IMG_SIZE // 2)

    # 计算光源坐标（光源位于图像顶部，可以在左右移动）
    light_x = int(center[0] + np.random.uniform(-LIGHT_RADIUS, LIGHT_RADIUS))
    light_y = center[1] - LIGHT_RADIUS

    # 计算摆锤尖端坐标
    pendulum_end_x = int(center[0] + PENDULUM_LENGTH * np.sin(phi2))
    pendulum_end_y = int(center[1] + PENDULUM_LENGTH * np.cos(phi2))

    # 计算摆锤底端坐标（假设摆锤长度为IMG_SIZE的1/4）
    pendulum_base_x = center[0]
    pendulum_base_y = center[1]

    # 构造图像
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 画光源
    draw.ellipse([light_x - 3, light_y - 3, light_x + 3, light_y + 3], fill=(255, 200, 0))

    # 画摆锤
    draw.line([(pendulum_base_x, pendulum_base_y), (pendulum_end_x, pendulum_end_y)], fill=(0, 0, 0), width=2)
    draw.ellipse([pendulum_end_x - 2, pendulum_end_y - 2, pendulum_end_x + 2, pendulum_end_y + 2], fill=(255, 0, 0))

    # 计算摆锤尖端的阴影
    dx_end = pendulum_end_x - light_x
    dy_end = pendulum_end_y - light_y
    if dy_end == 0:
        dy_end = 1e-5  # 避免除以0

    t_end = (IMG_SIZE - 1 - light_y) / dy_end
    shadow_end_x = light_x + t_end * dx_end
    shadow_end_y = IMG_SIZE - 2

    # 计算摆锤底端的阴影
    dx_base = pendulum_base_x - light_x
    dy_base = pendulum_base_y - light_y
    if dy_base == 0:
        dy_base = 1e-5  # 避免除以0

    t_base = (IMG_SIZE - 2 - light_y) / dy_base
    shadow_base_x = light_x + t_base * dx_base
    shadow_base_y = IMG_SIZE - 2

    # 确保投影点在图像范围内
    shadow_end_x = np.clip(shadow_end_x, 0, IMG_SIZE - 2)
    shadow_base_x = np.clip(shadow_base_x, 0, IMG_SIZE - 2)

    # 计算阴影长度
    shadow_len = np.abs(shadow_end_x - shadow_base_x)

    # 画阴影线
    draw.line([(shadow_end_x, shadow_end_y), (shadow_base_x, shadow_base_y)], fill=(100, 100, 100), width=2)
    labels = [light_x / IMG_SIZE, phi2, (shadow_end_x + shadow_base_x) / (2 * IMG_SIZE), shadow_len / IMG_SIZE]

    return img, labels


with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "light_x", "pendulum_angle", "shadow_pos", "shadow_len"])

    for i in range(NUM_SAMPLES):
        phi1 = np.random.uniform(-np.pi/4, np.pi/4)
        phi2 = np.random.uniform(-np.pi/4, np.pi/4)
        img, labels = generate_pendulum_image(phi1, phi2)

        filename = f"pendulum_{i:05d}.png"
        img.save(os.path.join(OUTPUT_DIR, filename))
        writer.writerow([filename] + labels)
