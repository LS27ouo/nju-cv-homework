import numpy as np
import cv2 as cv
import dlib
import math
from PIL import Image, ImageEnhance

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
# 设置数据源
glasspath = 'raw/glass2.png'
headwearpath = 'raw/headwear.jpg'
videopath='raw/06.MOV'


# 图像旋转，保持原来大小
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue=white
    return cv.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


# 磨皮功能
def mopi(frame):
    blur = cv.bilateralFilter(frame, 31, 75, 75)
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    # 融合
    img_add = cv.addWeighted(frame, alpha, blur, beta, gamma)

    img_add = Image.fromarray(cv.cvtColor(img_add, cv.COLOR_BGR2RGB))

    enh_sha = ImageEnhance.Sharpness(img_add)
    sharpness = 1.5
    # 增强
    image_sharped = enh_sha.enhance(sharpness)
    # 锐化
    enh_con = ImageEnhance.Contrast(image_sharped)
    contrast = 1.15
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted = cv.cvtColor(np.asarray(image_contrasted), cv.COLOR_RGB2BGR)
    return image_contrasted


# 计算旋转角度
def cntDegree(landmarks):
    # 角度旋转，通过计算两个眼角和水平方向的夹角来旋转眼镜
    sx = landmarks[36][0, 0] - landmarks[45][0, 0]
    sy = landmarks[36][0, 1] - landmarks[45][0, 1]
    # 夹角正切值
    r = sy / sx
    # 求正切角,弧度转为度
    degree = math.degrees(math.atan(r))
    return degree


# 装饰
def decorate(src, pos_left, pos_right, center, frame, degree, mixtype):
    size = src.shape
    length = pos_right[0] - pos_left[0]
    width = int(size[0] / (size[1] / length))
    src = cv.resize(src, (length, width), interpolation=cv.INTER_CUBIC)
    # 调用旋转方法
    src = rotate_bound(src, degree)
    # mask处理，去掉旋转后的无关区域，初始化一个全0mask，用或运算处理mask
    src_mask = 255 * np.ones(src.shape, src.dtype)
    # src_mask = np.zeros(src.shape, src.dtype)
    src_mask = cv.bitwise_or(src, src_mask)
    # 泊松融合
    if mixtype==0:
        frame = cv.seamlessClone(src, frame, src_mask, center, cv.MIXED_CLONE)
    return frame


def addHeadwear(frame,rect,src):
    x0, y0, width_face, height_face = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
    height_hat, width_hat = src.shape[0], src.shape[1]
    imgComposeSizeH = int(height_hat / width_hat * width_face)
    # if imgComposeSizeH > (y0 - 20):
    #     imgComposeSizeH = (y0 - 20)
    imgComposeSize = cv.resize(src, (width_face, imgComposeSizeH), interpolation=cv.INTER_NEAREST)
    top = (y0 - 20 - imgComposeSizeH)
    if top <= 0:
        top = 0
    height_src_new, width_src_new = imgComposeSize.shape[0], imgComposeSize.shape[1]
    small_img_src = frame[top:top + height_src_new, x0:x0 + width_src_new]
    small_img_src_gray = cv.cvtColor(imgComposeSize, cv.COLOR_RGB2GRAY)
    ret, mask_src = cv.threshold(small_img_src_gray, 10, 255, cv.THRESH_BINARY)
    mask_src_inv = cv.bitwise_not(mask_src)
    img1_bg = cv.bitwise_and(small_img_src, small_img_src, mask=mask_src_inv)
    img2_fg = cv.bitwise_and(imgComposeSize, imgComposeSize, mask=mask_src)
    dst = cv.add(img1_bg, img2_fg)
    # 将局部区域贴合到原始图像上
    frame[top:top + height_src_new, x0:x0 + width_src_new] = dst
    return frame


def detect_face(camera_idx):
    # camera_idx: 电脑自带摄像头或者usb摄像头
    # cv.namedWindow('detect')
    cap = cv.VideoCapture(camera_idx)
    # 2. 获取图像的属性（宽和高，）,并将其转换为整数
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter('outpy.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))  # 保存视频
    while cap.isOpened():
        # cv.namedWindow('detect', cv.WINDOW_AUTOSIZE)
        ok, frame = cap.read()
        # 为摄像头的时候，翻转画面
        if camera_idx == 0 or camera_idx == 1:
            frame = cv.flip(frame, 1, dst=None)
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        frame = mopi(frame)
        output = frame
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rects[i]).parts()])
            # 脸轮廓：1~17
            # 眉毛：18~22, 23~27
            # 鼻梁：28~31
            # 鼻子：31~36
            # 眼睛：37~42, 43~48
            # 嘴唇：49~68
            # frame = utils.face_thin(frame, [landmarks])
            # 左眼角和右眼角的位置

            # 图片旋转角度
            degree = cntDegree(landmarks)

            pos_left = (landmarks[0][0, 0], landmarks[36][0, 1])
            pos_right = (landmarks[16][0, 0], landmarks[45][0, 1])
            face_center = (landmarks[27][0, 0], landmarks[27][0, 1])

            glassSrc = cv.imread(glasspath)
            headwearSrc = cv.imread(headwearpath)
            frame = decorate(glassSrc, pos_left, pos_right, face_center,frame, degree, 0)
            frame = addHeadwear(frame, rects[i], headwearSrc)
        # cv.imshow('detect', frame)
        out.write(frame)
        c = cv.waitKey(10)
        if c & 0xFF == ord('q'):
            print("视频未生成完，非正常退出！")
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # 把图片该成透明背景
    # img = Image.open('raw/5.png')  # 读取照片
    # img = img.convert("RGBA")  # 转换格式，确保像素包含alpha通道
    # width, height = img.size  # 长度和宽度
    # for i in range(0, width):  # 遍历所有长度的点
    #     for j in range(0, height):  # 遍历所有宽度的点
    #         data = img.getpixel((i, j))  # 获取一个像素
    #         if (data.count(100) == 3 or data.count(255)==4) :  # RGBA都是255，改成透明色
    #             img.putpixel((i, j), (255, 255, 255, 0))
    # img.save('raw/7.png')  # 保存图片
    print("正在生成视频")
    detect_face(videopath)
    # detect_face(0)  # 前置摄像头
    print("生成结束")
