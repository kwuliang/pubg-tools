import cv2
import numpy 
import os

def extract_features(image_path,search_region_coords=None, threshold=200):
    
    image = cv2.imread(image_path)
    if search_region_coords is not None:
        x1, x2, y1, y2 = search_region_coords
        height, width = image.shape[:2]
        x_start, x_end = int(x1 * width), int(x2 * width)
        y_start, y_end = int(y1 * height), int(y2 * height)
        image = image[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(thresholded, None)
    return keypoints, descriptors, image

def match_features(descriptors1, descriptors2):
    # 使用 FLANN 匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 需要绘制好的匹配，只保留好的匹配结果
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def main(image_path1, image_path2):
    
    keypoints1, descriptors1, image1 = extract_features(image_path1)
    keypoints2, descriptors2, image2 = extract_features(image_path2,[0.71,0.835,0.87,0.975])

    good_matches = match_features(descriptors1, descriptors2)

    # 绘制匹配结果
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)
    cv2.imshow("Matches", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Number of good matches: {len(good_matches)}")  # 打印好的匹配数目

def compare_images(base_image_path, directory_path):
    keypoints1, descriptors1, _ = extract_features(base_image_path,[0.71,0.835,0.87,0.975])
    max_matches = 0
    best_match_image = None
    best_keypoints = None

    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        if image_path == base_image_path or not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        keypoints2, descriptors2, _ = extract_features(image_path)
        good_matches = match_features(descriptors1, descriptors2)
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_match_image = filename
            best_keypoints= keypoints2

    return best_match_image, max_matches,keypoints1,best_keypoints

def draw(image1, keypoints1, image2, keypoints2, good_matches):
    # 绘制匹配结果
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)
    cv2.imshow("Matches", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Number of good matches: {len(good_matches)}")  # 打印好的匹配数目


if __name__ == "__main__":
    
    
    # main('../guns/AK47.jpg', '../testimgs/guns_test_ace32.jpg')
    
    for imgs in os.listdir("../testimgs"):
    
        base_image= os.path.join("../testimgs",imgs)
        best_match_image,good_matches,_,_=compare_images(base_image,"../guns")
        print("\n==> source image: ",base_image)
        print(f"Mostmatches: {best_match_image}")  # 打印好的匹配数目
        print(f"Number of good matches: {good_matches}")  # 打印好的匹配数目
    