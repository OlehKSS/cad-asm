import os

import cv2
import matplotlib.pyplot as plt

from ashapes import align, match_average_shape
from ashapes.utils import no_points


# Finds average shape, builds active shapes model and tests it
# Path to hand data
path_shapes = "./data/hand/shapes/shapes.txt"
path_shapes_norm = "./data/hand/shapes/shapes_norm.txt"
path_images = "./data/hand/images"
test_img_name = 4

test_img_path = os.path.join(path_images, str(test_img_name).zfill(4) + ".jpg")

shapes_norm = align(path_shapes, path_images, path_shapes_norm, test_img_name)
new_shape = match_average_shape(shapes_norm, test_img_path, plot_results=False)

img = cv2.imread(test_img_path, 0)
plt.imshow(img, cmap="gray")
plt.plot(new_shape[:no_points], new_shape[no_points:])
plt.show()