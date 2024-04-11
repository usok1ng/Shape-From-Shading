# Minimization Approach

import numpy as np
import cv2
from function import estimate
import matplotlib.pyplot as plt

E = cv2.imread('sphere.jpg', cv2.IMREAD_GRAYSCALE)
E = E[::2, ::2]
E = E.astype(np.float64)
E /= np.max(E)

albedo, I, _, _ = estimate(E)

M, N = E.shape

p = np.zeros((M, N))
q = np.zeros((M, N))
p_ = np.zeros((M, N))
q_ = np.zeros((M, N))
R = np.zeros((M, N))

lamda = 1000
maxIter = 100

w = 0.25 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
x, y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
wx = (2 * np.pi * x) / M
wy = (2 * np.pi * y) / N

for k in range(maxIter):
    p_ = cv2.filter2D(p, -1, w, borderType=cv2.BORDER_REPLICATE)
    q_ = cv2.filter2D(q, -1, w, borderType=cv2.BORDER_REPLICATE)

    R = (albedo * (-I[0] * p - I[1] * q + I[2])) / np.sqrt(1 + p ** 2 + q ** 2)

    pq = (1 + p ** 2 + q ** 2)
    dR_dp = (-albedo * I[0] / (pq ** (1 / 2))) + (-I[0] * albedo * p - I[1] * albedo * q + I[2] * albedo) * (
            -1 * p * (pq ** (-3 / 2)))
    dR_dq = (-albedo * I[1] / (pq ** (1 / 2))) + (-I[0] * albedo * p - I[1] * albedo * q + I[2] * albedo) * (
            -1 * q * (pq ** (-3 / 2)))

    p = p_ + (1 / (4 * lamda)) * (E - R) * dR_dp
    q = q_ + (1 / (4 * lamda)) * (E - R) * dR_dq

    Cp = np.fft.fft2(p)
    Cq = np.fft.fft2(q)

    C = -1j * (wx * Cp + wy * Cq) / (wx ** 2 + wy ** 2)

    Z = np.abs(np.fft.ifft2(C))

    p = np.fft.ifft2(1j * wx * C).real
    q = np.fft.ifft2(1j * wy * C).real

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, Z, cmap='gray')

plt.show()