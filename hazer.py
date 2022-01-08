import math
import fractions
import numpy as np

# This lib is build to detecting haze img

def getDarkness(rgbArray):
    return np.min(rgbArray, axis=2)

def getBrightness(rgbArray):
    return np.max(rgbArray, axis=2)

def getHazeFactor(rgbArray, mu = 5.1, nu = 2.9, sigma = 0.2461):

    dI = getDarkness(rgbArray)

    bI = getBrightness(rgbArray)

    cI = bI - dI

    d = np.mean(dI)

    b = np.mean(bI)

    c = b - d

    # 分数
    lamda = fractions.Fraction(1, 3)

    bmax = np.max(bI)

    A0 = lamda * bmax + (1 - lamda) * b

    x1 = (A0 - d) / A0
    x2 = c / A0

    w = math.exp(-0.5 * (mu * x1 + nu * x2) + sigma)

    return w

