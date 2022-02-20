import cv2
import numpy as np


def post_handler(img):
    img_t = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_t)

    # 增加图像亮度
    v1 = np.clip(cv2.add(1 * v, 15), 0, 255)

    img1 = np.uint8(cv2.merge((h, s, v1)))
    img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)

    cv2.imshow("result", img1)
    cv2.imwrite("/Users/joey777210/Documents/paper_img/0011_result_clear.jpg", img1)

    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread("/Users/joey777210/Documents/paper_img/0011_result.jpg")
    post_handler(img)
