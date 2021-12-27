import numpy as np
import wls


def regularization(row, col, transmission_estimation, img_hazy_corrected, n_points, air, cluster_Points, cluster):
    # Regularization
    # Apply lower bound from the image (Eqs. (13-14)
    trans_lower_bound = np.zeros([row, col], dtype=float)

    for i in range(row):
        for j in range(col):
            m = min(img_hazy_corrected[i][j][0] / air[0], img_hazy_corrected[i][j][1] / air[1],
                    img_hazy_corrected[i][j][2] / air[2])
            trans_lower_bound[i][j] = 1 - m

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
