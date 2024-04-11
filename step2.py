import numpy as np
import cv2
from math import acos, atan, sin, cos, pi

E = cv2.imread('sphere.jpg', cv2.IMREAD_GRAYSCALE)
E = E / np.max(E)
Mu1 = np.mean(E)
Mu2 = np.mean(np.square(E))

print("Average of the image brightness:", Mu1)
print("Average of the image brightness square:", Mu2)

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

print("gamma:", gamma)
print("albedo:", albedo)
print("slant:", slant)
print("tilt:", tilt)
print("Estimated illumination direction:", I)