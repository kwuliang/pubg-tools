import cv2
import os

# 初始化 SIFT 算子
sift = cv2.SIFT_create()

# 读取枪械模板图像的特征
gun_features = {}
guns_path = '../guns'
for filename in os.listdir(guns_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        gun_name = filename.split('.')[0]
        gun_image = cv2.imread(os.path.join(guns_path, filename), cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(gun_image, None)
        gun_features[gun_name] = descriptors

# 读取游戏截屏图像，并裁剪出右下角的区域
screenshot = cv2.imread('../testimgs/AK47.jpg', cv2.IMREAD_GRAYSCALE)  # 请替换为游戏截屏的文件名
h, w = screenshot.shape[:2]
cropped_screenshot = screenshot[int(0.75 * h):, int(0.75 * w):]  # 裁剪出右下角区域

# 提取右下角区域的特征
keypoints, descriptors = sift.detectAndCompute(cropped_screenshot, None)

# 定义 FLANN 特征匹配器
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 计算与每个枪械模板的匹配程度
best_match_name = None
max_good_matches = 0
for gun_name, gun_descriptors in gun_features.items():
    matches = flann.knnMatch(gun_descriptors, descriptors, k=2)
    # 使用 Lowe's ratio test 筛选好的匹配点
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    if len(good_matches) > max_good_matches:
        max_good_matches = len(good_matches)
        best_match_name = gun_name

# 输出匹配到的枪械名字
print(f"识别到的枪械是: {best_match_name}")
