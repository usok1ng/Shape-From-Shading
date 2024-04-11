import numpy as np
import matplotlib.pyplot as plt
import cv2

r = 50
x = np.arange(-1.5 * r, 1.5 * r, 0.1)
y = np.arange(-1.5 * r, 1.5 * r, 0.1)
x, y = np.meshgrid(x, y) 

albedo = 0.5
I = np.array([0.2, 0, 0.98])

# surface partial derivative at each point in the pixel grid
p = -x / np.sqrt(r ** 2 - (x ** 2 + y ** 2))
q = -y / np.sqrt(r ** 2 - (x ** 2 + y ** 2))

# image brightness at each point in the pixel grid
rmap = (albedo * (-I[0] * p - I[1] * q + I[2])) / np.sqrt(1 + p ** 2 + q ** 2)

mask = r ** 2 - (x ** 2 + y ** 2) >= 0

rmap = rmap * mask

E = np.maximum(0, rmap)

plt.imshow(E, cmap='gray')
plt.axis('off')
plt.show()

E = (E * 255).astype(np.uint8)
cropped_E = E[200:1300, 200:1300]
cv2.imwrite('sphere.jpg', cropped_E)