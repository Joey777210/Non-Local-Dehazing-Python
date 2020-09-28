import cv2
import numpy as np
import math
import sys
import kdtree
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

file_path = "/home/joey/Documents/pumpkins_input.png"
filter_size = 15
p = 0


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


def non_local_transmission(img, air, gamma=1):
    ## find airlight first (same method with DCP)
    img_hazy_corrected = np.power(img, gamma)
    # img = img / 255
    dist_from_airlight = getDistAirlight(img_hazy_corrected, air)
    row, col, n_colors = img.shape

    # Calculate radius(Eq.(5))
    # 3 - dimentional
    radius = np.sqrt(np.sum(dist_from_airlight ** 2, axis=2))

    # Cluster the pixels to haze-lines
    # Use a KD-tree impementation for fast clustering according to their angles
    dist_sphere_radius = np.reshape(radius, [col * row], order='F')

    # 3-di To 2-di      (col*row, 3)
    dist_unit_radius = np.reshape(dist_from_airlight, [col * row, n_colors], order='F')

    dist_norm = np.sqrt(np.sum(dist_unit_radius ** 2, axis=1))

    for i in range(len(dist_unit_radius)):
        dist_unit_radius[i] = dist_unit_radius[i] / dist_norm[i]

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
    # save pixel cluster to which point (save index)
    cluster_Points = np.zeros(row * col, dtype=np.int)

    for r in range(len(dist_unit_radius)):
        kdNode = mdl.search_knn(dist_unit_radius[r], 1)
        findPosition(kdNode[0][0].data, dist_sphere_radius[r], cluster, points, r, cluster_Points)

    # how to use the data in kdNode
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
    np.reshape(maxRadius, [row, col], order='F')

    # Initial Transmission
    # save maxRadius to all pixels
    dist_sphere_maxRadius = np.zeros(row * col, np.float)
    for i in range(row * col):
        index = cluster_Points[i]
        dist_sphere_maxRadius[i] = maxRadius[index]
    radius_new = np.reshape(dist_sphere_maxRadius, [row, col], order='F')

    transmission_estimation = radius / radius_new

    # Limit the transmission to the range [trans_min, 1] for numerical stability
    trans_min = 0.1

    # for i in range(row):
    #     for j in range(col):
    #         transmission_estimation = min(max(transmission_estimation[i][j], trans_min), 1)
    transmission_estimation = np.minimum(np.maximum(transmission_estimation, trans_min), 1)
    # ## Regularization
    # # Apply lower bound from the image (Eqs. (13-14)
    trans_lower_bound = np.zeros([row, col], dtype=float)

    for i in range(row):
        for j in range(col):
            m = min(img_hazy_corrected[i][j][0] / air[0], img_hazy_corrected[i][j][1] / air[1],
                    img_hazy_corrected[i][j][2] / air[2])
            trans_lower_bound[i][j] = 1 - m + p

    transmission_estimation = np.maximum(transmission_estimation, trans_lower_bound)

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

    max_radius_std = np.max(radius_std)
    temp = radius_std / max_radius_std - 0.1
    temp = np.where(temp > 0.001, temp, 0.001) * 3

    radius_reliability = np.where(temp > 1, 1, temp)

    temp2 = np.where(bin_count_map > 1, 1, bin_count_map / 50)

    data_term_weight = temp2 * radius_reliability
    lambd = 0.1

    trans = np.reshape(transmission_estimation, (row, col), order='F')

    transmission = wls_filter(trans, data_term_weight, img_hazy_corrected.astype(np.float32), lambd)

    return transmission


def wls_filter(in_, data_term_weight, guidance, lambda_=0.05, small_num=0.00001):
    h, w, _ = guidance.shape
    k = h * w

    guidance = cv2.cvtColor(guidance, cv2.COLOR_RGB2GRAY).tolist()

    # Compute affinities between adjacent pixels based on gradients of guidance
    dy = np.diff(guidance, axis=0)
    dy = - lambda_ / (np.abs(dy) ** 2 + small_num)
    dy = np.pad(dy, ([0, 1], [0, 0]), 'constant', constant_values=0)
    dy = dy.flatten('F').T

    dx = np.diff(guidance, axis=1)
    dx = -lambda_ / (np.abs(dx) ** 2 + small_num)
    dx = np.pad(dx, ([0, 0], [0, 1]), 'constant', constant_values=0)
    dx = dx.flatten(order='F').T

    B = np.vstack((dx, dy))
    d = [-h, -1]
    tmp = sparse.spdiags(B, d, k, k)
    # row vector
    ea = dx
    temp = [dx]
    we = np.pad(temp, ([0, 0], [h, 0]))[0]
    we = we[0:len(we) - h]

    # row vector
    so = dy
    temp = [dy]
    no = np.pad(temp, ([0, 0], [1, 0]))[0]
    no = no[0:len(no) - 1]

    # row vector
    D = -(ea + we + so + no)
    Asmoothness = tmp + tmp.T + sparse.spdiags(D, 0, k, k)
    # Normalize data weight
    data_weight = data_term_weight - np.min(data_term_weight)

    data_weight = data_weight / (np.max(data_weight) + small_num)

    # Make sure we have a boundary condition for the top line:
    # It will be the minimum of the transmission in each column
    # With reliability 0.8
    reliability_mask = np.where(data_weight[0] < 0.6, 1, 0)
    in_row1 = np.min(in_, axis=0)
    # print(reliability_mask)
    for i in range(w):
        if reliability_mask[i] == 1:
            data_weight[0][i] = 0.8

    for i in range(w):
        if reliability_mask[i] == 1:
            in_[0][i] = in_row1[i]

    Adata = sparse.spdiags(data_weight.flatten(), 0, k, k)

    A = Asmoothness + Adata

    b = Adata * in_.flatten(order='F').T

    X = spsolve(A, b)

    out = np.reshape(X, [h, w], order='F')
    np.savetxt("./out.txt", out, fmt="%0.5f", delimiter="\t")
    return out


# cluster into 1000length arr
def findPosition(kdNode, radius, cluster, points, r, cluster_Points):
    for i in range(len(points)):
        if (points[i][0] == kdNode[0]) and (points[i][1] == kdNode[1]) and (points[i][2] == kdNode[2]):
            cluster[i].append(radius)
            cluster_Points[r] = i
            break


def non_local_dehazing(img, transmission_estimission, air):
    img = img / 255
    air = air
    row, col, _ = img.shape
    # print(trans)
    result = np.empty_like(img, dtype=float)
    for i in range(3):
        result[:, :, i] = ((img[:, :, i] - air[i]) / transmission_estimission) + air[i]
    return result


def dehaze(img, img_gray, transmission_estimission, air):
    h, w, n_colors = img.shape
    img_dehazed = np.zeros((h, w, n_colors), dtype=float)
    leave_haze = 1.06
    for color_idx in range(3):
        img_dehazed[:, :, color_idx] = (img_gray[:, :, color_idx] - (1 - leave_haze * transmission_estimission) * air[
            color_idx]) / np.maximum(transmission_estimission, 0.1)

    img_dehazed = np.where(img_dehazed > 1, 1, img_dehazed)
    img_dehazed = np.where(img_dehazed < 0, 0, img_dehazed)
    img_dehazed = np.power(img_dehazed, 1 / 1)
    adj_percent = [0.005, 0.995]

    # img_dehazed = adjust(img_dehazed, adj_percent)
    # img_dehazed = (img_dehazed * 255).astype(np.uint8)
    # print(img_dehazed)
    return img_dehazed


def adjust(img_dehazed, adj_percent):
    minn = np.min(img_dehazed)
    img_dehazed = img_dehazed - minn
    maxx = np.max(img_dehazed)
    img_dehazed = img_dehazed / maxx
    return img_dehazed
    # contrast_limit = stretchlim


def main():
    img = cv2.imread(file_path)
    cv2.imshow("input_image", img)
    img_gray = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    dark = dark_channel(img, filter_size)
    air = air_light(img, dark)
    air = air[0] / 255

    # trans DCP
    trans = get_trans(img, dark / 255, air[0])
    # cv2.imshow("trans_DCP", trans)

    # trans Nonl-Local
    transmission_estimission = non_local_transmission(img_gray, air)

    clear_img = dehaze(img, img_gray, transmission_estimission, air)
    clear_img2 = non_local_dehazing(img, transmission_estimission, air)
    # clear_img2 = non_local_dehazing(img, trans, air)
    cv2.imshow("result", clear_img)
    cv2.imshow("result2", clear_img2)
    cv2.imshow("non-local transmission", transmission_estimission)


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    sys.setrecursionlimit(100000)
    main()
    cv2.waitKey(0)
