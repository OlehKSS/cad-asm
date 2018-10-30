import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt

from PCA import pca

# Path to aligned data
path = "./data/hand/shapes/shapes_norm.txt"
path_image = "./data/hand/images/0000.jpg"

shapes_norm = np.loadtxt(path, np.float32)
img = cv2.imread(path_image,0)

no_points = 56
no_shapes = 40

#Image gradient
sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
grad_mag = np.sqrt(sobelx**2 + sobely**2) / 1448
# cv2.imshow('image',img)
# cv2.imshow('mag',grad_mag)
# cv2.waitKey(0)


# Initialise the shape parameters, b, to zero (the mean shape)
b = np.zeros((112, 1))

# Generate the model point positions using x = x_bar + P * b
x_bar, largest_evals, P = pca(shapes_norm)
new_x_bar = 30000*x_bar+ 300


# should be done with protocol 1 pg 9
# Norm Calculation
x_barx = new_x_bar[:no_points]
x_bary = new_x_bar[no_points:]

ux = x_barx[1] - x_barx[0]
uy = x_bary[1] - x_bary[0]

nx = -uy / sqrt(ux * ux + uy * uy)
ny = ux / sqrt(ux * ux + uy * uy)

print(nx, ny)

# x' = x + t*nx
x = x_barx[0] + 20 * nx
y = x_bary[0] + 20 * ny

print(x, y)

plt.imshow(img, cmap = "gray")
plt.plot(new_x_bar[:no_points], new_x_bar[no_points:], 'o', color='#F82A80')
plt.plot((x, x_barx[0]), (y, x_bary[0]))
plt.show()