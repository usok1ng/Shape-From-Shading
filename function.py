import numpy as np
from math import acos, atan, sin, cos, pi

def estimate(E):
    Mu1 = np.mean(E)
    Mu2 = np.mean(np.square(E))

    Ex, Ey = np.gradient(E)

    Exy = np.sqrt(Ex ** 2 + Ey ** 2)
    nEx = Ex / (Exy + np.finfo(float).eps)  
    nEy = Ey / (Exy + np.finfo(float).eps)

    avgEx = np.mean(Ex)
    avgEy = np.mean(Ey)

    gamma = np.sqrt((6 * (pi ** 2) * Mu2) - (48 * (Mu1 ** 2)))
    albedo = gamma / pi

    slant = acos((4 * Mu1) / gamma)

    tilt = atan(avgEy / avgEx)
    if tilt < 0:
        tilt = tilt + pi

    I = [cos(tilt) * sin(slant), sin(tilt) * sin(slant), cos(slant)]

    return albedo, I, slant, tilt