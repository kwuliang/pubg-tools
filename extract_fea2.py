import cv2
import numpy as np

def extract_white_object_features(image):
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 设定白色的阈值范围
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    # 创建白色物体的掩码
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 找到白色物体的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 绘制所有轮廓
    contour_image = np.copy(image)
    cv2.drawContours(contour_image, [max_contour], -1, (0, 255, 0), 2)
    
    # 计算合并后的轮廓的特征
    # 计算轮廓的面积
    area = cv2.contourArea(max_contour)
    # 计算轮廓的周长
    perimeter = cv2.arcLength(max_contour, True)
    # 获取轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(max_contour)
    # 计算外接矩形的宽高比
    aspect_ratio = float(w) / h if h != 0 else 0
    
    features = {
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio
    }
    
    return features, contour_image

# 读取图像
image = cv2.imread('./guns/AK47.jpg')
# 提取白色物体的特征和绘制轮廓的图像
white_object_features, contour_image = extract_white_object_features(image)

# 显示绘制轮廓的图像
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 打印特征
print("Feature of Merged Contours:")
for key, value in white_object_features.items():
    print(f"{key}: {value}")
