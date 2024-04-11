# Implementation of Shah's Approach

import cv2
import numpy as np
from function import estimate
from math import cos, sin, tan
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

E = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
E = E[::2, ::2]
E = E.astype(np.float64)
E /= np.max(E)

albedo, I, slant, tilt = estimate(E)

M, N = E.shape

x, y = np.meshgrid(np.arange(1, N+1), np.arange(1, M+1))

p = np.zeros_like(E)
q = np.zeros_like(E)
Z = np.zeros_like(E)
Z_x = np.zeros_like(E)
Z_y = np.zeros_like(E)

maxIter = 100

ix = cos(tilt) * tan(slant)
iy = sin(tilt) * tan(slant)

for k in range(maxIter):
    R = (cos(slant) + p * cos(tilt) * sin(slant) + q * sin(tilt) * sin(slant)) / np.sqrt(1 + p ** 2 + q ** 2)
    R = np.maximum(0, R)
    f = E - R
    df_dZ = (p + q) * (ix * p + iy * q + 1) / (np.sqrt((1 + p ** 2 + q ** 2) ** 3) * np.sqrt(1 + ix ** 2 + iy ** 2)) - (ix + iy) / (np.sqrt(1 + p ** 2 + q ** 2) * np.sqrt(1 + ix ** 2 + iy **2))

    Z = Z - f / (df_dZ + np.finfo(float).eps)
    
    Z_x[2:M, :] = Z[1:M-1, :]
    Z_y[:, 2:N] = Z[:, 1:N-1]

    p = Z - Z_x
    q = Z - Z_y

window_size = (3, 3)
Z = median_filter(np.abs(Z), size=window_size)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.plot_surface(x, y, Z, cmap='gray')
plt.show()