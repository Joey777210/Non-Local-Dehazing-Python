import cv2
import numpy as np
import kdtree
import utils
import regularization
import hazer

file_path = "/Users/joey777210/Documents/paper_img/test2.jpg"
filter_size = 15
# 修正参数, 每个簇都应该有一个，初始化为随机数，根据启发式算法调节
p = 0
n_points = 1000

# 缓存聚类结果
max_radius = np.zeros(n_points, dtype=np.float)
# 进入计数
count = 0


def adjust(max_radius, chromosome):
    max_radius_adjusted = []
    if len(chromosome) == 0:
        return max_radius.copy()

    for i in range(n_points):
        max_radius_adjusted.append(max_radius[i] + chromosome[i])

    return max_radius_adjusted


def non_local_transmission(img, air, chromosome, gamma=1):
    # find airlight first (same method with DCP)
    img_hazy_corrected = np.power(img, gamma)

    # img = img / 255
    dist_from_airlight = utils.getDistAirlight(img_hazy_corrected, air)

    row, col, n_colors = img.shape

    # Calculate radius(Eq.(5))
    # 3 - dimentional
    radius = np.sqrt(np.sum(dist_from_airlight ** 2, axis=2))

    # ----------------------缓存聚类结果---------------------------
    global max_radius, cluster_points, cluster_radius
    global count

    count += 1

    # 首次进入，聚类，并缓存结果(可复用，不影响最终结果)
    if count == 1:
        # Cluster the pixels to haze-lines
        # Use a KD-tree impementation for fast clustering according to their angles
        dist_sphere_radius = np.reshape(radius, [col * row], order='F')

        # 3-di To 2-di      (col*row, 3)
        dist_unit_radius = np.reshape(dist_from_airlight, [col * row, n_colors], order='F')

        dist_norm = np.sqrt(np.sum(dist_unit_radius ** 2, axis=1))

        for i in range(len(dist_unit_radius)):
            dist_unit_radius[i] = dist_unit_radius[i] / dist_norm[i]

        tr_file = "./TR" + str(n_points) + ".txt"
        points = np.loadtxt(tr_file).tolist()
        mdl = kdtree.create(points)

        # lines stores cluster result
        cluster_radius = [[]] * n_points

        # save pixel cluster to which point (save index)
        cluster_points = np.zeros(row * col, dtype=np.int)

        for pos in range(len(dist_unit_radius)):
            kd_node = mdl.search_knn(dist_unit_radius[pos], 1)
            cluster_center = kd_node[0][0].data
            findPosition(cluster_center, dist_sphere_radius[pos], cluster_radius, points, pos, cluster_points)

        # Estimating Initial Transmission
        # Estimate radius as the maximal radius in each haze-line (Eq. (11))
        for i in range(n_points):
            # find max radius
            if len(cluster_radius[i]) == 0:
                max_radius[i] = 0
                continue

            max_radius[i] = max(cluster_radius[i])

    # ----------------------缓存聚类结果---------------------------

    # 添加修正
    max_radius_adjusted = adjust(max_radius, chromosome)

    # Initial Transmission
    # save maxRadius to all pixels
    dist_sphere_max_radius = np.zeros(row * col, np.float)
    for i in range(row * col):
        index = cluster_points[i]
        dist_sphere_max_radius[i] = max_radius_adjusted[index]
    radius_max = np.reshape(dist_sphere_max_radius, [row, col], order='F')

    transmission_estimation = radius / radius_max

    # Limit the transmission to the range [trans_min, 1] for numerical stability
    trans_min = 0.1

    transmission_estimation = np.minimum(np.maximum(transmission_estimation, trans_min), 1)

    transmission = regularization.regularization(row, col, transmission_estimation, img_hazy_corrected, n_points, air,
                                                 cluster_points, cluster_radius)
    return transmission


# cluster into 1000length arr
def findPosition(center, radius, cluster_radius, points, pos, cluster_points):
    for i in range(len(points)):
        if (points[i][0] == center[0]) and (points[i][1] == center[1]) and (points[i][2] == center[2]):
            cluster_radius[i].append(radius)
            cluster_points[pos] = i
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


def cal_w(chromosome):
    img = cv2.imread(file_path)
    img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    dark = utils.dark_channel(img, filter_size)
    air = utils.air_light(img, dark)
    air = air[0] / 255

    # Non-Local transmission
    transmission_estimission = non_local_transmission(img_norm, air, chromosome)

    clear_img = dehaze(img, img_norm, transmission_estimission, air)
    w = hazer.getHazeFactor(clear_img)
    return w


def get_pic(chromosome):
    img = cv2.imread(file_path)
    img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)  # Convert to normalized floating point
    dark = utils.dark_channel(img, filter_size)
    air = utils.air_light(img, dark)
    air = air[0] / 255

    # Non-Local transmission
    transmission_estimission = non_local_transmission(img_norm, air, [])
    transmission_estimission_adjust = non_local_transmission(img_norm, air, chromosome)

    clear_img = dehaze(img, img_norm, transmission_estimission, air)
    clear_img_adjust = dehaze(img, img_norm, transmission_estimission_adjust, air)

    cv2.imshow("before", clear_img)
    cv2.imshow("after", clear_img_adjust)
    cv2.waitKey(0)
