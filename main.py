
import cv2
import numpy as np
import math

file_path = "/home/joey/Documents/fog150.png"
filter_size = 7

# alter Rectangle system to Spherical coordinate system
def get_sphere(rect, img):
    row, col, _ = img.shape
    r = [0] * (row * col)
    phi = [0] * (row * col)
    theta = [0] * (row * col)

    for i in range(row) :
        for j in range(col) :
            cur = i * col + j
            # cal r
            r[cur] = math.sqrt(math.pow(rect[0][cur], 2) + math.pow(rect[1][cur], 2) + math.pow(rect[2][cur], 2))
            # cal phi
            if rect[0][cur] == 0 :
                phi[cur] = (180/math.pi) * math.atan(float(rect[1][cur])/0.000001)
            else :
                phi[cur] = (180/math.pi) * math.atan(float(rect[1][cur]) / float(rect[0][cur]))
            # cal theta
            if r[cur] == 0 :
                theta[cur] = math.acos(rect[2][cur] / (r[cur] + 0.000001)) * 180 / math.pi
            else :
                theta[cur] = math.acos(rect[2][cur] / r[cur]) * 180 / math.pi
    sphere = [r, phi, theta]
    return sphere

# calculate r g b value - air.(r/g/b) respectivly
# and save them in an array named rectangle
# It means a rectangle coordinate system which airlight as the original point
def get_rectangle(img, air):
    row, col, _ = img.shape

    red = [0] * (row * col)
    green = [0] * (row * col)
    blue = [0] * (row * col)

    for i in range(row) :
        for j in range(col) :
            cur = i * col + j
            red[cur] = img[i, j, 2] - int(air[2])
            green[cur] = img[i, j, 1] - air[1]
            blue[cur] = img[i, j, 0] - air[0]

    rect = [red, green, blue]
    return rect


def build_sph_sys(img, air):
    # first, get all destance between rgb values and air
    rect = get_rectangle(img, air)

    # second, alter Rectangle system to Spherical coordinate system
    sphere = get_sphere(rect, img)

def dark_channel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    cv2.imshow('dark', dark)
    return dark

def air_light(im, dark):
    [h,w] = im.shape[:2]
    image_size = h*w
    numpx = int(max(math.floor(image_size/1000),1))
    darkvec = dark.reshape(image_size,1);
    imvec = im.reshape(image_size,3);

    indices = darkvec.argsort();
    indices = indices[image_size-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def non_local_dehazing(img):
    # find airlight first (same method with DCP)
    dark = dark_channel(img, filter_size)
    air = air_light(img, dark)
    air = air[0]
    # for i in range(3) :
        # air[i] = int(air[i])

    # build Spherical coordinate system
    build_sph_sys(img, air)


def main():
    img = cv2.imread(file_path)
    cv2.imshow("input_image", img)
    non_local_dehazing(img)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)