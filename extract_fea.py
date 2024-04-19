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
    
    # 初始化特征列表和绘制轮廓的图像
    features = []
    contour_image = np.copy(image)
    
    # 对每个轮廓提取特征并绘制在图像上

    count =0
    for contour in contours:
        count +=1 
        if count >4:
            break
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 计算外接矩形的宽高比
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # 将特征添加到特征列表
        features.append({
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio
        })
        
        # 在原始图像上绘制轮廓
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
    
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
for idx, feature in enumerate(white_object_features):
    print(f"Feature {idx+1}: {feature}")

