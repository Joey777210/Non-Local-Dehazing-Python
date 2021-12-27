import cv2
import numpy as np
import math

# calculate r g b value - air.(r/g/b) respectivly
# and save them in an array named rectangle
# It means a rectangle coordinate system which airlight as the original point
def getDistAirlight(img, air):
    row, col, n_colors = img.shape

    dist_from_airlight = np.zeros((row, col, n_colors), dtype=np.float)
    for color in range(n_colors):
        dist_from_airlight[:, :, color] = img[:, :, color] - air[color]

    return dist_from_airlight


def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def air_light(im, dark):
    [h, w] = im.shape[:2]
    image_size = h * w
    numpx = int(max(math.floor(image_size / 1000), 1))
    darkvec = dark.reshape(image_size, 1)
    imvec = im.reshape(image_size, 3)

    indices = darkvec.argsort()
    indices = indices[image_size - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A
