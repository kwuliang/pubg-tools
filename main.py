import utils as util
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


def startdeal(screen_images,imagename="",save_dir="output",guns_dir= "./guns"):
     # 遍历枪械文件夹
    if guns_dir=="" :
        return 
    
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
    guns_name_predict, best_contour,best_rect,max_similarity,_ = util.find_most_similar(guns_images_map, screen_images, search_region_coords)
    
    
    if max_similarity == -1:
        return
    print("Most similar to:", guns_name_predict," max_similarity: ",max_similarity)
    
    gunsinfo = 'guns_name = "' + guns_name_predict + '"'
    
    updateLua("guninfo.lua",gunsinfo)
    
    if save_dir != "":
        
        if imagename == "":
            imagename ="test"
        # 在原图上绘制最相似目标的轮廓
        util.draw_contour(screen_images, best_contour, (x_start, y_start))

        # 在原图上绘制最相似目标的轮廓
        util.draw_rectangle(screen_images, (x_start+best_rect[0],y_start+best_rect[1],best_rect[2],best_rect[3]))
        
        imagename = imagename+"_"+guns_name_predict+".jpg"
        impath = os.path.join(save_dir,imagename)
        
        cv2.imwrite(impath, screen_images)
        
        # == 展示出来
        
        # cv2.imshow("Matched Result with Contour", screen_images)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
def test_video():
    file_path = 'D:\\wuliang\\Aworkspace\\pyw\\video\\guns_test.mp4'

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
            startdeal(frame,imagename,save_dir="output")
            print(count)

    vc.release()
    
    
def test_imgdir():
    dirpath= "testimgs"
    imglist = os.listdir(dirpath)
    for imgs in imglist:
        imgpath =os.path.join(dirpath,imgs)
        screen_images = cv2.imread(imgpath)
        startdeal(screen_images,imgs.split(".")[0],"output")
    
def test_contours():
    
    # mask = util.preprocess_image_for_white_objects(search_region)
    # imgs= "./2kguns/AK47.jpg"
    # search_region = cv2.imread(imgs)
    
    imgs ="../pubgimgs/guns_test_30.jpg"
    search_image = cv2.imread(imgs)
    
    height, width = search_image.shape[:2]
    # 定义裁剪的比例范围
    x_start, x_end = int(0.75 * width), int(0.835 * width)
    y_start, y_end = int(0.87 * height), int(0.975 * height)
    
    search_region_coords = (x_start, y_start, x_end - x_start, y_end - y_start)
    x, y, w, h = search_region_coords
    search_region = search_image[y:y+h, x:x+w]
    
    mask = util.preprocess_image_for_white_objects(search_region)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    util.showContours(search_region,contours,(0,0))
    
    
 
if __name__ == "__main__":
    
    test_imgdir()
    # test_video()
    # test_contours()
    
        
     