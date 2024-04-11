# Implementation of Pentland's approach

import cv2
import numpy as np
from function import estimate
from math import cos, sin
import matplotlib.pyplot as plt

E = cv2.imread('sphere.jpg', cv2.IMREAD_GRAYSCALE)
E = E[::2, ::2]
E = E.astype(np.float64)
E /= np.max(E)

albedo, I, slant, tilt = estimate(E)

Fe = np.fft.fft2(E)

M, N = E.shape
x, y = np.meshgrid(np.arange(1, N+1), np.arange(1, M+1))
wx = (2 * np.pi * x) / M
wy = (2 * np.pi * y) / N

Fz = Fe / (-1j * wx * cos(tilt) * sin(slant) - 1j * wy * sin(tilt) * sin(slant))

Z = np.abs(np.fft.ifft2(Fz))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, Z, cmap='gray')

plt.show()