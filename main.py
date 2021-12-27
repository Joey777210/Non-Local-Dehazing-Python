import cv2
import numpy as np
import sys
import kdtree
import utils
import wls

file_path = "../Pics/city_input.png"
filter_size = 15
# 修正参数, 每个簇都应该有一个，初始化为随机数，根据启发式算法调节
p = 0


def non_local_transmission(img, air, gamma=1):
    # find airlight first (same method with DCP)
    img_hazy_corrected = np.power(img, gamma)
    # img = img / 255
    dist_from_airlight = utils.getDistAirlight(img_hazy_corrected, air)
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

    # Estimating Initial Transmission
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

    transmission = wls.wls_filter(trans, data_term_weight, img_hazy_corrected.astype(np.float32), lambd)

    return transmission


# cluster into 1000length arr
def findPosition(kdNode, radius, cluster, points, r, cluster_Points):
    for i in range(len(points)):
        if (points[i][0] == kdNode[0]) and (points[i][1] == kdNode[1]) and (points[i][2] == kdNode[2]):
            cluster[i].append(radius)
            cluster_Points[r] = i
            break


def dehaze(img, img_norm, transmission_estimission, air):
    h, w, n_colors = img.shape
    img_dehazed = np.zeros((h, w, n_colors), dtype=float)
    leave_haze = 1.06
    for color_idx in range(3):
        img_dehazed[:, :, color_idx] = (img_norm[:, :, color_idx] - (1 - leave_haze * transmission_estimission) * air[
            color_idx]) / np.maximum(transmission_estimission, 0.1)

    img_dehazed = np.where(img_dehazed > 1, 1, img_dehazed)
    img_dehazed = np.where(img_dehazed < 0, 0, img_dehazed)
    img_dehazed = np.power(img_dehazed, 1 / 1)
    adj_percent = [0.005, 0.995]

    # img_dehazed = adjust(img_dehazed, adj_percent)
    # img_dehazed = (img_dehazed * 255).astype(np.uint8)
    # print(img_dehazed)
    return img_dehazed


def main():
    img = cv2.imread(file_path)
    cv2.imshow("input_image", img)
    img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    dark = utils.dark_channel(img, filter_size)
    air = utils.air_light(img, dark)
    air = air[0] / 255

    # Non-Local transmission
    transmission_estimission = non_local_transmission(img_norm, air)

    clear_img = dehaze(img, img_norm, transmission_estimission, air)

    cv2.imshow("result", clear_img)
    cv2.imshow("non-local transmission", transmission_estimission)


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    sys.setrecursionlimit(100000)
    main()
    cv2.waitKey(0)
