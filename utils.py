import cv2
import numpy as np


def preprocess_image_for_white_objects(image):
    """ 提取亮白色物体 """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 设定白色的阈值范围，调整这些阈值以只包含亮白色区域
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # 使用膨胀操作让物体区域更加明显
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask



def find_most_similar(target_images, search_image, search_region_coords):
    """ 在给定区域内找到与目标图像最相似的图像，并返回最相似目标的轮廓信息和矩形框 """
    x, y, w, h = search_region_coords
    search_region = search_image[y:y+h, x:x+w]
    mask = preprocess_image_for_white_objects(search_region)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = None
    max_similarity = -1
    guns_name = None
    best_contour = None
    best_rect = None

    for name, target_img in target_images.items():
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            candidate = search_region[y:y+h, x:x+w]
            candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)

            shape_similarity = cv2.matchShapes(target_gray, candidate_gray, cv2.CONTOURS_MATCH_I1, 0)
            # print("similarity ==> ",shape_similarity)
            if shape_similarity < max_similarity or max_similarity == -1:
                max_similarity = shape_similarity
                best_match = candidate
                guns_name = name
                best_contour = cnt
                best_rect = ( x, y, w, h)
                 

    return guns_name, best_contour,best_rect,max_similarity,best_match


def draw_contour(image, contour, offset, color=(0, 255, 0), thickness=2):
    """ 在图像上绘制轮廓 """
    # 因为轮廓是在搜索区域中找到的，需要加上偏移量来画在原始图像上
    shifted_contour = contour.copy()
    shifted_contour[:, :, 0] += offset[0]  # X offset
    shifted_contour[:, :, 1] += offset[1]  # Y offset
    cv2.drawContours(image, [shifted_contour], -1, color, thickness)
    
    
def draw_rectangle(image, rect, color=(0, 255, 0), thickness=2):
    """ 在图像上绘制矩形 """
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)