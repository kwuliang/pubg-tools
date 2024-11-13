import cv2
import os

def recognise(fea_path=None, dest_img=None):
    if fea_path is None or dest_img is None:
        return None
    
    # 初始化 SIFT 算子
    sift = cv2.SIFT_create()

    # 读取枪械模板图像的特征
    gun_features = {}
    guns_path = fea_path
    for filename in os.listdir(guns_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            gun_name = filename.split('.')[0]
            gun_image = cv2.imread(os.path.join(guns_path, filename), cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(gun_image, None)
            if descriptors is not None and len(descriptors) >= 2:  # 检查描述符是否至少有 2 个
                gun_features[gun_name] = descriptors

    # 读取游戏截屏图像，并裁剪出右下角的区域
    screenshot = cv2.imread(dest_img, cv2.IMREAD_GRAYSCALE)
    h, w = screenshot.shape[:2]
    cropped_screenshot = screenshot[int(0.75 * h):, int(0.75 * w):]  # 裁剪出右下角区域

    # 提取右下角区域的特征
    keypoints, descriptors = sift.detectAndCompute(cropped_screenshot, None)
    if descriptors is None or len(descriptors) < 2:  # 如果描述符少于 2 个，直接返回 None
        return None

    # 定义 FLANN 特征匹配器
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 计算与每个枪械模板的匹配程度
    best_match_name = None
    max_good_matches = 0
    for gun_name, gun_descriptors in gun_features.items():
        # 确保 gun_descriptors 和 descriptors 都有足够的特征进行匹配
        if len(gun_descriptors) >= 2 and len(descriptors) >= 2:
            matches = flann.knnMatch(gun_descriptors, descriptors, k=2)
            # 使用 Lowe's ratio test 筛选好的匹配点
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_name = gun_name

    return best_match_name


def main():
    fea_path = '../guns'  # 枪械模板图片文件夹路径
    screenshots_folder = '../testimgs'  # 游戏截屏存放文件夹路径

    # 遍历游戏截屏文件夹
    for screenshot_filename in os.listdir(screenshots_folder):
        if screenshot_filename.endswith('.jpg') or screenshot_filename.endswith('.png'):
            screenshot_path = os.path.join(screenshots_folder, screenshot_filename)
            gun_name = recognise(fea_path=fea_path, dest_img=screenshot_path)
            print(f"图片: {screenshot_filename}，识别到的枪械: {gun_name}")

if __name__ == '__main__':
    main()