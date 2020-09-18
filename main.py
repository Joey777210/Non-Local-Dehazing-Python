import cv2
import numpy as np
import math
import sys
import kdtree
import scipy.sparse as sparse
import scipy.sparse.linalg as sl
from scipy import optimize

file_path = "/home/joey/Documents/tiananmen.png"
filter_size = 15
p = 0


# calculate r g b value - air.(r/g/b) respectivly
# and save them in an array named rectangle
# It means a rectangle coordinate system which airlight as the original point
def getDistAirlight(img, air):
    row, col, deep = img.shape

    dist_from_airlight = np.zeros((row, col, deep), dtype=np.float)
    for color in range(deep):
        dist_from_airlight[:, :, color] = img[:, :, color] - air[color]

    return dist_from_airlight


def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    cv2.imshow('dark', dark)
    return dark


def get_trans(img, dark, atom, w=0.95):
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
    for i in range(n_points):
        cluster[i] = []
    # save pixel in img cluster to which point (index)
    cluster_Points = np.zeros(row * col, dtype=np.int)

    for r in range(len(dist_unit_radius)):
        kdNode = mdl.search_knn(dist_unit_radius[r], 1)
        findPosition(kdNode[0][0].data, dist_sphere_radius[r], cluster, points, r, cluster_Points)

    # how to use the data
    # print(lines[0][0][0].data[0])

    ## Estimating Initial Transmission
    # Estimate radius as the maximal radius in each haze-line (Eq. (11))
    maxRadius = np.zeros(row * col, dtype=np.float)
    for i in range(n_points):
        # find max radius
        maxR = 0
        for j in range(len(cluster[i])):
            maxR = max(maxR, cluster[i][j])
        maxRadius[i] = maxR

    # Initial Transmission
    # save maxRadius to all pixels
    dist_sphere_maxRadius = np.zeros(row * col, np.float)
    for i in range(row * col):
        index = cluster_Points[i]
        dist_sphere_maxRadius[i] = maxRadius[index]
    transmission_estimation = dist_sphere_radius / dist_sphere_maxRadius

    # Limit the transmission to the range [trans_min, 1] for numerical stability
    trans_min = 0.1

    # ## Regularization
    # # Apply lower bound from the image (Eqs. (13-14)
    trans_lower_bound = np.zeros((row * col), dtype=float)

    for i in range(row):
        for j in range(col):
            m = min(img[i][j][0] / air[0], img[i][j][1] / air[1], img[i][j][2] / air[2])
            trans_lower_bound[i * col + j] = 1 - m + p

    for i in range(len(transmission_estimation)):
        transmission_estimation[i] = min(max(transmission_estimation[i], trans_lower_bound[i], trans_min), 1)

    # Solve optimization problem (Eq. (15))
    # find bin counts for reliability - small bins (#pixels<50) do not comply with
    # the model assumptions and should be disregarded
    bin_count = np.zeros(n_points, int)
    for index in cluster_Points:
        bin_count[index] += 1

    bin_count_map = np.zeros((row, col), np.int)
    radius_std = np.zeros((row, col), np.float)

    K_std = np.zeros(n_points, np.float)
    for i in range(n_points):
        if len(cluster[i]) > 0:
            K_std[i] = np.std(cluster[i])

    for i in range(row * col):
        index = cluster_Points[i]
        bin_count_map[int(i / col)][int(i % col)] = bin_count[index]
        radius_std[int(i / col)][int(i % col)] = K_std[index]

    #####
    max_radius_std = np.max(radius_std)
    temp = radius_std / max_radius_std - 0.1
    temp = np.where(temp > 0.001, temp, 0.001) * 3

    radius_reliability = np.where(temp > 1, 1, temp)

    temp2 = bin_count_map / 50
    temp2 = np.where(temp > 1, 1, temp)
    data_term_weight = np.multiply(temp, radius_reliability)
    lambd = 0.1

    trans = np.reshape(transmission_estimation, (row, col))
    transmission = wls_filter(trans, data_term_weight, img, lambd)

    return transmission_estimation


def wls_filter(in_, data_term_weight, guidance, lambda_=0.1, alpha=2, small_num=1e-4):
    h, w, _= guidance.shape
    in_ = np.reshape(in_, (h, w))
    k = h * w
    guidance = cv2.cvtColor(guidance, cv2.COLOR_RGB2GRAY).tolist()

    # Compute affinities between adjacent pixels based on gradients of guidance
    dy = np.diff(guidance)
    dy = -lambda_ / (np.abs(dy) ** 2 + small_num)
    dy = np.pad(dy, ([0,0], [0,1]), 'edge')
    dy = dy.flatten('F').T

    dx = np.diff(guidance, 1, 0)
    dx = -lambda_ / (np.abs(dx) ** 2 + small_num)
    dx = np.pad(dx, ([0,1], [0,0]), 'edge')
    dx = dx.flatten().T

    B = np.vstack((dx, dy))

    tmp = sparse.spdiags(B, [-h, -1], k, k)

    # row vector
    ea = dx
    temp = [dx]
    we = np.pad(temp,([0,0], [h,0]))[0]
    we = we[0:len(we) - h]

    # row vector
    so = dy
    temp = [dy]
    no = np.pad(temp,([0,0], [1,0]))[0]
    no = no[0:len(no) - 1]

    # row vector
    D = -(ea + we + so + no)

    Asmoothness = tmp + tmp + sparse.spdiags(D, 0, k, k)

    # *******************************************
    # Normalize data weight
    data_weight = data_term_weight - np.min(data_term_weight)
    data_weight = 1 * data_weight / (np.max(data_weight) + small_num)

    # Make sure we have a boundary condition for the top line:
    # It will be the minimum of the transmission in each column
    # With reliability 0.8
    first_row = data_weight[1]
    reliability_mask = np.where(data_weight < 0.6, 1, 0)
    in_row1 = np.min(in_, axis=0)

    data_weight[1, reliability_mask] = 0.8
    # temp indicates in_row1(reliability_mask)
    # print(reliability_mask)
    # temp = []
    # for i in range(len(reliability_mask)) :
    #     index = reliability_mask[i]
    #     print(index)
        # temp[i] = in_row1[index]
        # print(temp[i])
        # print("--------------")

    for i in range(h) :
        for j in range(w) :
            if reliability_mask[i][j] == 1 :
                in_[i][j] = in_row1[j]

    Adata = sparse.spdiags(data_weight.flatten(), 0, k, k)
    A = Asmoothness + Adata
    b = Adata * in_.flatten()
    # x, info = sl.cg(A=A, b=b)

    # x = optimize.leastsq(A, b)
    # x = optimize.nnls(A.tocsr(), b)
    X = sparse.linalg.inv(A)
    # X = np.linalg.inv(A.toarray()).dot(b)
    # x = x.reshape(in_.shape)
    out = np.reshape(X, (h, w))
    print(out)
    return out


def process_difference_operator(difference_operator, lambda_, alpha, epsilon):
    difference_operator = -lambda_ / (epsilon + (np.absolute(difference_operator) ** alpha))
    return difference_operator


# cluster into 1000length arr
def findPosition(kdNode, radius, cluster, points, r, cluster_Points):
    for i in range(len(points)):
        if (points[i][0] == kdNode[0]) and (points[i][1] == kdNode[1]) and (points[i][2] == kdNode[2]):
            cluster[i].append(radius)
            cluster_Points[r] = i
            break


def non_local_dehazing(img, transmission_estimission, air):
    row, col, _ = img.shape
    trans = np.reshape(transmission_estimission, (row, col))
    # print(trans)
    cv2.imshow("estim", trans)
    result = np.empty_like(img, dtype=float)
    for i in range(3):
        # result[:, :, i] = (img[:, :, i]/255 - air[i]/255) / trans + air[i]/255
        result[:, :, i] = (img[:, :, i] - air[i]) / 255 / (trans) + air[i] / 255
    return result


def main():
    img = cv2.imread(file_path)
    cv2.imshow("input_image", img)
    dark = dark_channel(img, filter_size)
    air = air_light(img, dark)
    air = air[0]

    trans = get_trans(img, dark / 255, air[0])
    cv2.imshow("trans_DCP", trans)
    transmission_estimission = non_local_transmission(img, air)
    # print("est")
    # print(transmission_estimission)
    clear_img = non_local_dehazing(img, transmission_estimission, air)
    # print("  ")
    # print("DCP")
    clear_img2 = non_local_dehazing(img, trans, air)
    cv2.imshow("result", clear_img)
    cv2.imshow("result2", clear_img2)


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    main()
    cv2.waitKey(0)
    # row
    # d = [[1,23,1,4]]
    # dy = np.pad(d, ([0,0], [1, 0]))
    # print(dy)
    # N = sparse.spdiags(d, [-1, 0], 4, 4)
    # # a = np.zeros((4, 4), dtype=np.int)
    # N = N.A
    # print(N)

