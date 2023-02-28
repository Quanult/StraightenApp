from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np



def make_mask(img):
    im0 = cv2.blur(img, (1,1))
    ret,thresh1 = cv2.threshold(im0, 2,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    bg= cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    mask = bg.clip(0,1)
    img_denoising = (img * mask)
    img_color = cv2.cvtColor(img_denoising, cv2.COLOR_GRAY2BGR)
    return img_color, bg

def convertImage(img):
    # img = Image.open(img_path)
    img = Image.fromarray(img)
    img = img.convert("RGBA")
    # print(type(img))
    # print(img.size)
    datas = img.getdata()
    newData = []
 
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
 
    img.putdata(newData)
    # img.save("./New.png", "PNG")
    img = np.asarray(img)
    # print(img.shape)
    # print("Successful")
    return img

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

def insert_img(img, img_background, x, y):
    
    bg = cv2.threshold(img, 254,255,cv2.THRESH_BINARY_INV)[1][:, :, 0]
    kernel = np.ones((5,5),np.uint8)
    bg= cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)
    # print(bg.shape)
    # cv2.imshow('   ', bg)
    # cv2.waitKey(0)
    for i in range(256):
        if np.argmax(bg[255-i]) != 0:
            y_buf = i
            break
    # print(y_buf)
    x_cor = int(x - img.shape[1]/2 )
    y_cor = int(y - (img.shape[0] - y_buf))


    alpha_mask = img[:, :, 3] / 255.0
    img_result = img_background[:, :, :3].copy()
    img_overlay = img[:, :, :3]
    overlay_image_alpha(img_result, img_overlay, x_cor, y_cor, alpha_mask)
    
    return img_result

def rewrite_image(chro_list, bg_img, class_box):
    # if len(bg_img.shape) == 3:
    #     bg_img = bg_img[:, :, 0]
    # white_bg = np.ones_like(bg_img) * 255
    for idx, chro_img in enumerate(chro_list):
        x, y, w, h = class_box[idx]
        img = convertImage(chro_img)
        if idx == 1:
            cv2.imwrite('test.jpg', img)
        bg_img = insert_img(img, bg_img, x + w/2, y + h)

    return bg_img


if __name__ == "__main__":
    # img = mpimg.imread('./Karyotype_image/M6Anstcopy.PNG')
    # imgplot = plt.imshow(img)
    # img = cv2.imread('./Karyotype_image/M6.A.nst.png')
    # print(img.shape)
    # img_bg = cv2.imread('img_result.jpg')
    # img = convertImage(img) 
    # im = insert_img(img,img_bg, 1200 , 300 ) (x, y)
    # Image.fromarray(im).save("img_result.jpg")
    
    img = cv2.imread('./test.jpg')
    img[100:110,100:110, : ] = 255
    cv2.imshow('    ', img)
    cv2.waitKey(0)

    








