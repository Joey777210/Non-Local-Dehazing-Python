import cv2
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

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
    return out
