
import numpy as np
import phasepack.phasecong as pc
import cv2
import os

def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg


def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def fsim(
    org_img: np.ndarray, pred_img: np.ndarray, T1: float = 0.85, T2: float = 160
) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)

    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.

    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.

    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.

    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.

    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(
            org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )
        pc2_2dim = pc(
            pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros(
            (pred_img.shape[0], pred_img.shape[1]), dtype=np.float64
        )
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def psnr(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Peek Signal to Noise Ratio, implemented as mean squared error converted to dB.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When using 12-bit imagery MaxP is 4095, for 8-bit imagery 255. For floating point imagery using values between
    0 and 1 (e.g. unscaled reflectance) the first logarithmic term can be dropped as it becomes 0
    """
    _assert_image_shapes_equal(org_img, pred_img, "PSNR")

    mse_bands = []
    for i in range(org_img.shape[2]):
        mse_bands.append(np.mean(np.square(org_img[:, :, i] - pred_img[:, :, i])))

    return 20 * np.log10(max_p) - 10.0 * np.log10(np.mean(mse_bands))


def read_image(path: str):
    return cv2.imread(path)


if __name__ == '__main__':
    org_dir_path = "/Users/joey777210/Documents/论文相关/paper_img/对比实验/定性分析/真实图像org/"
    pred_dir_path = "/Users/joey777210/Documents/论文相关/paper_img/对比实验/定性分析/真实图像/"

    for filename in os.listdir(pred_dir_path):  # listdir的参数是文件夹的路径
        if filename.startswith('.'): continue

        pred_img_path = pred_dir_path + filename
        pred_img = read_image(pred_img_path)
        print("-----------", filename, "------------")
        prefix = filename.split('_')[0]
        org_img = None
        for org_filename in os.listdir(org_dir_path):
            if org_filename[:len(org_filename) - 4][4:] == prefix:
                org_img = read_image(org_dir_path + org_filename)

        if pred_img.shape[0] == 520:
            org_img = cv2.resize(org_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        psnr_out_value = float(psnr(org_img, pred_img))
        fsim_out_value = float(fsim(org_img, pred_img))
        print(filename + "\tpsnr_out_value\t", psnr_out_value)
        print(filename + "\tfsim_out_value\t", fsim_out_value)

