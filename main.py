import cv2
import numpy as np
import math
import sys
import kdtree
import numpy_groupies as npg
import unittest
import cmath

file_path = "/home/joey/Documents/color2.jpg"
filter_size = 7


# ---------------------------------------------definition of kdtree------------------------------------------------------------------------
class KdtreeNode:
    x = 0
    y = 0
    depth = 0
    SpherePoint = []
    left = None
    right = None

    def __init__(self, x: int, y: int, depth: int, SpherePoint: [3]):
        self.x = x
        self.y = y
        self.depth = depth
        self.SpherePoint = SpherePoint

    def setLeft(self, left):
        self.left = left

    def setRight(self, right):
        self.right = right


def NewKdtreeNode(r, phi, theta, row, col, dep):
    sphePoint = [r, phi, theta]
    node = KdtreeNode(row, col, dep, sphePoint)
    node.setLeft(None)
    node.setRight(None)
    return node


def InsertNode(root, r, phi, theta, x, y, dep):
    if root is None:
        return KdtreeNode(x, y, dep, [r, phi, theta])
    if dep % 2 == 0:
        if phi < root.SpherePoint[1]:
            root.left = InsertNode(root.left, r, phi, theta, x, y, dep + 1)
        else:
            root.right = InsertNode(root.right, r, phi, theta, x, y, dep + 1)

    if dep % 2 != 0:
        if theta < root.SpherePoint[2]:  
            root.left = InsertNode(root.left, r, phi, theta, x, y, dep + 1)
        else:
            root.right = InsertNode(root.right, r, phi, theta, x, y, dep + 1)
    return root


def build_kdtree(sphere, img):
    root = None
    row, col, _ = img.shape
    print(len(sphere[2]))
    print((row - 1) * col + col - 1)
    for i in range(row):
        for j in range(col):
            cur = i * col + j
            root = InsertNode(root, sphere[0][cur], sphere[1][cur], sphere[2][cur], i, j, 0)

    return root


# Root First
def KdtreeErgodic(root: KdtreeNode):
    result = []
    ptr = root
    stack = []
    while (not ptr is None) or len(stack) != 0:
        while not ptr is None:
            result.append(ptr)
            stack.append(ptr)
            ptr = ptr.left
        if len(stack) != 0:
            ptr = stack.pop()
            ptr = ptr.right

    return result


def getDepth(root):
    depth = 0
    ergodic = KdtreeErgodic(root)
    for node in ergodic:
        if node.depth > depth:
            depth = node.depth
    return depth


# ---------------------------------------------end kdtree----------------------------------------------------------------------------------

def clusterHazeline(root, dep):
    ergodic = KdtreeErgodic(root)
    hazeline = []

    for node in ergodic:
        if node.depth == dep:
            hazeline.append(node)
    return hazeline


# alter Rectangle system to Spherical coordinate system
def get_sphere(rect, img):
    row, col, _ = img.shape
    r = [0] * (row * col)
    phi = [0] * (row * col)
    theta = [0] * (row * col)

    for i in range(row):
        for j in range(col):
            cur = i * col + j
            # cal r
            r[cur] = math.sqrt(math.pow(rect[0][cur], 2) + math.pow(rect[1][cur], 2) + math.pow(rect[2][cur], 2))
            # cal phi
            if rect[0][cur] == 0:
                phi[cur] = (180 / math.pi) * math.atan(float(rect[1][cur]) / 0.000001)
            else:
                phi[cur] = (180 / math.pi) * math.atan(float(rect[1][cur]) / float(rect[0][cur]))
            # cal theta
            if r[cur] == 0:
                theta[cur] = math.acos(rect[2][cur] / (r[cur] + 0.000001)) * 180 / math.pi
            else:
                theta[cur] = math.acos(rect[2][cur] / r[cur]) * 180 / math.pi
    sphere = [r, phi, theta]
    return sphere


# calculate r g b value - air.(r/g/b) respectivly
# and save them in an array named rectangle
# It means a rectangle coordinate system which airlight as the original point
def getDistAirlight(img, air):
    row, col, deep = img.shape

    dist_from_airlight = np.zeros((row, col, deep), dtype=np.float)
    for color in range(deep) :
        dist_from_airlight[:,:,color] = img[:,:,color] - air[color]

    return dist_from_airlight


def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    cv2.imshow('dark', dark)
    return dark


def air_light(im, dark):
    [h, w] = im.shape[:2]
    image_size = h * w
    numpx = int(max(math.floor(image_size / 1000), 1))
    darkvec = dark.reshape(image_size, 1);
    imvec = im.reshape(image_size, 3);

    indices = darkvec.argsort();
    indices = indices[image_size - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def sampleLayerNum(root, samplePoint):
    ergodic = KdtreeErgodic(root)

    for i in range(getDepth(root)):
        num = 0
        for node in ergodic:
            if node.depth == i:
                num += 1
        if num > samplePoint:
            return i


def non_local_dehazing(img, AirlightAdjust, samplePoint):
    ## find airlight first (same method with DCP)
    dark = dark_channel(img, filter_size)
    air = air_light(img, dark)
    air = air[0]
    # for i in range(len(air)):
    #     air[i] = air[i] / 255

    dist_from_airlight = getDistAirlight(img, air)
    row, col, deep = img.shape

    # 3 - dimentional
    # # radius = np.sqrt(np.power(dist_from_airlight[:,:,0], 2), np.power(dist_from_airlight[:,:,1], 2), np.power(dist_from_airlight[:,:,2], 2))
    radius = np.sqrt(np.sum(dist_from_airlight ** 2, axis=2))
    dist_sphere_radius = np.reshape(radius, [col * row])

    # 3-di To 2-di
    dist_unit_radius = np.reshape(dist_from_airlight, [col * row, deep])

    dist = np.sum(dist_unit_radius ** 2, axis=1)

    dist_norm = np.sqrt(dist)

    for i in range(len(dist_unit_radius)):
        for j in range(3):
            dist_unit_radius[i][j] = dist_unit_radius[i][j] / dist_norm[i]

    n_points = 1000

    file_path = "./TR" + str(n_points) + ".txt"
    points = np.loadtxt(file_path).tolist()

    mdl = kdtree.create(points)
    # lines stores cluster result
    # [(<KDNode - [0.6256, 0.5636, 0.5394]>, 0.003957329357625813)]
    #       cluster node from points              distance
    cluster = [[]] * n_points
    for i in range(n_points) :
        cluster[i] = []

    for r in range(len(dist_unit_radius)) :
        kdNode = mdl.search_knn(dist_unit_radius[r], 1)
        findPosition(kdNode[0][0].data, dist_sphere_radius[r], cluster, points)
    print(cluster[15][0])
    # how to use the data
    # print(lines[0][0][0].data[0])

    ## Estimating Initial Transmission
    # Estimate radius as the maximal radius in each haze-line (Eq. (11))
    maxRadius = [[]] * n_points
    for i in range(n_points) :
        # find max radius
        maxR = 0
        for j in range(len(cluster[i])) :
            maxR = max(maxR, cluster[i][j])
        maxRadius[i] = [maxR]
    print(maxRadius)


# cluster into 1000length arr
def findPosition(kdNode, radius, cluster, points) :
    for i in range(len(points)) :
        if (points[i][0] == kdNode[0]) and (points[i][1] == kdNode[1]) and (points[i][2] == kdNode[2]) :
            cluster[i].append(radius)
            break

def main():
    img = cv2.imread(file_path)
    cv2.imshow("input_image", img)
    non_local_dehazing(img, 1, 1000)




if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    main()
    cv2.waitKey(0)

