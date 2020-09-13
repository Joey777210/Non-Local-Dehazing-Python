import cv2
import numpy as np
import math
import sys
import kdtree

file_path = "/home/joey/Documents/tiananmen.png"
filter_size = 15
p = 0

# calculate r g b value - air.(r/g/b) respectivly
# and save them in an array named rectangle
# It means a rectangle coordinate system which airlight as the original point
def getDistAirlight(img, air):
    row, col, deep = img.shape

    dist_from_airlight = np.zeros((row, col, deep), dtype=np.float)
    for color in range(deep) :
        dist_from_airlight[:,:,color] = img[:,:,color] - air[color]

    return dist_from_airlight


def dark_channel(im, sz) :
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    cv2.imshow('dark', dark)
    return dark


def get_trans(img, dark, atom, w = 0.95):
    x = img / atom
    p = 0.12
    t = 1 - w * dark
    return t


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


def non_local_transmission(img, air):
    ## find airlight first (same method with DCP)

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
    # save pixel in img cluster to which point (index)
    cluster_Points = np.zeros(row * col, dtype=np.int)

    for r in range(len(dist_unit_radius)) :
        kdNode = mdl.search_knn(dist_unit_radius[r], 1)
        findPosition(kdNode[0][0].data, dist_sphere_radius[r], cluster, points, r, cluster_Points)

    # how to use the data
    # print(lines[0][0][0].data[0])

    ## Estimating Initial Transmission
    # Estimate radius as the maximal radius in each haze-line (Eq. (11))
    maxRadius = np.zeros(row * col, dtype=np.float)
    for i in range(n_points) :
        # find max radius
        maxR = 0
        for j in range(len(cluster[i])) :
            maxR = max(maxR, cluster[i][j])
        maxRadius[i] = maxR

    # Initial Transmission
    # save maxRadius to all pixels
    dist_sphere_maxRadius = np.zeros(row * col, np.float)
    for i in range(row * col) :
        index = cluster_Points[i]
        dist_sphere_maxRadius[i] = maxRadius[index]
    transmission_estimation = dist_sphere_radius / dist_sphere_maxRadius

    # Limit the transmission to the range [trans_min, 1] for numerical stability
    trans_min = 0.1

    # ## Regularization
    # # Apply lower bound from the image (Eqs. (13-14)
    trans_lower_bound = np.zeros((row * col), dtype=float)

    for i in range(row) :
        for j in range(col) :
            m = min(img[i][j][0]/air[0], img[i][j][1]/air[1], img[i][j][2]/air[2])
            trans_lower_bound[i * col + j] = 1 - m + p

    for i in range(len(transmission_estimation)) :
        transmission_estimation[i] = min(max(transmission_estimation[i], trans_lower_bound[i], trans_min), 1)

    # Solve optimization problem (Eq. (15))
    # find bin counts for reliability - small bins (#pixels<50) do not comply with
    # the model assumptions and should be disregarded
    bin_count = np.zeros(n_points, int)
    for index in cluster_Points :
        bin_count[index] += 1

    bin_count_map = np.zeros((row, col), np.int)
    for i in range(row * col) :
        index = cluster_Points[i]
        bin_count_map[int(i/col)][int(i%col)] = bin_count[index]
    ##############################################################################################


    return transmission_estimation

# cluster into 1000length arr
def findPosition(kdNode, radius, cluster, points, r, cluster_Points) :
    for i in range(len(points)) :
        if (points[i][0] == kdNode[0]) and (points[i][1] == kdNode[1]) and (points[i][2] == kdNode[2]) :
            cluster[i].append(radius)
            cluster_Points[r] = i
            break

def non_local_dehazing(img, transmission_estimission, air) :
    row, col, _ = img.shape
    trans = np.reshape(transmission_estimission, (row, col))
    print(trans)
    cv2.imshow("estim", trans)
    result = np.empty_like(img, dtype=float)
    for i in range(3):
        # result[:, :, i] = (img[:, :, i]/255 - air[i]/255) / trans + air[i]/255
        result[:, :, i] = (img[:, :, i] - air[i])/255 / (trans) + air[i]/255
    return result


def main():
    img = cv2.imread(file_path)
    cv2.imshow("input_image", img)
    dark = dark_channel(img, filter_size)
    air = air_light(img, dark)
    air = air[0]

    trans = get_trans(img, dark/255, air[0])
    cv2.imshow("trans_DCP", trans)
    transmission_estimission = non_local_transmission(img, air)
    print("est")
    print(transmission_estimission)
    clear_img = non_local_dehazing(img, transmission_estimission, air)
    print("  ")
    print("DCP")
    clear_img2 = non_local_dehazing(img, trans, air)
    cv2.imshow("result", clear_img)
    cv2.imshow("result2", clear_img2)


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    main()
    cv2.waitKey(0)

