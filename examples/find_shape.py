import os

import cv2
import matplotlib.pyplot as plt

from ashapes import align, match_average_shape
from ashapes.utils import no_points


# Finds average shape, builds active shapes model and tests it

# Set paths to shapes.txt file, images folder, and specify test image
path_shapes = "./data/hand/shapes/shapes.txt"
path_images = "./data/hand/images"
test_img_name = 4

test_img_path = os.path.join(path_images, str(test_img_name).zfill(4) + ".jpg")

shapes_norm, _ = align(path_shapes, path_images, test_img_name)
new_shape = match_average_shape(shapes_norm, test_img_path, plot_results=True)

img = cv2.imread(test_img_path, 0)
plt.imshow(img, cmap="gray")
plt.title("Final Result")
plt.plot(new_shape[:no_points], new_shape[no_points:])
plt.show()