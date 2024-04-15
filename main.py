import utils
import cv2
import os
 
def test():
    screen_images = cv2.imread("./testimages/m762-2.jpg")
    startdeal(screen_images,"test.jpg")
    

def updateLua(luapath, info):
    # 打开 test.lua 文件以覆盖写入模式
    with open(luapath, 'w') as file:
    # 将新内容写入文件
        file.write(info)


def startdeal(screen_images,imagename="",save_dir=""):
     # 遍历枪械文件夹
    guns_dir = "./guns"
    guns_imgs = os.listdir(guns_dir)
    
    # 获取输入（屏幕截图）
    print(guns_imgs)
    
     
    
    guns_images_map={}
    for guns in guns_imgs:
    
        rpath = os.path.join(guns_dir,guns)
         
        guns_name = os.path.basename(rpath).split(".")[0]
        
        images_fea = cv2.imread(rpath)
        guns_images_map[guns_name]= images_fea
        
        
    # screen_images = cv2.imread(screen_images_path)
    # 获取图像尺寸
    height, width = screen_images.shape[:2]

    # 定义裁剪的比例范围
    x_start, x_end = int(0.75 * width), int(0.835 * width)
    y_start, y_end = int(0.87 * height), int(0.975 * height)
    search_region_coords = (x_start, y_start, x_end - x_start, y_end - y_start)
    
    # 找到最相似的目标和轮廓
    guns_name_predict, best_contour,best_rect,max_similarity,_ = utils.find_most_similar(guns_images_map, screen_images, search_region_coords)
    
    
    print("Most similar to:", guns_name_predict," max_similarity: ",max_similarity)
    
    gunsinfo = 'guns_name = "' + guns_name_predict + '"'
    
    updateLua("guninfo.lua",gunsinfo)
    
    if save_dir != "":
        
        if imagename == "":
            imagename ="test"
        # 在原图上绘制最相似目标的轮廓
        utils.draw_contour(screen_images, best_contour, (x_start, y_start))

        # 在原图上绘制最相似目标的轮廓
        utils.draw_rectangle(screen_images, (x_start+best_rect[0],y_start+best_rect[1],best_rect[2],best_rect[3]))
        
        imagename = imagename+"_"+guns_name_predict+".jpg"
        impath = os.path.join(save_dir,imagename)
        
        cv2.imwrite(impath, screen_images)
        
        # == 展示出来
        
        # cv2.imshow("Matched Result with Contour", screen_images)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    
    file_path = 'testimages/video-2.mp4'

    vc = cv2.VideoCapture(file_path)  # import video files
    # determine whether to open normally
    if vc.isOpened():
        ret, frame = vc.read()
    else:
        ret = False
    
    count = 0  # count the number of pictures

    # loop read video frame
    while ret:
        ret, frame = vc.read()
        count += 1
        if count %30==0:
            imagename = os.path.basename(file_path).split(".")[0]+str(count)
            startdeal(frame,imagename,save_dir="")
            print(count)

    vc.release()
    
        
     