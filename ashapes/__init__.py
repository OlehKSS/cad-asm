"""Active Shapes Method in Python"""
import os

from align_shapes import align
from matching import match_average_shape, match_average_shape_pyr


# Path to hand data
path_shapes = "./data/hand/shapes/shapes.txt"
path_shapes_norm = "./data/hand/shapes/shapes_norm.txt"
path_images = "./data/hand/images"
test_img_name = 4

test_img_path = os.path.join(path_images, str(test_img_name).zfill(4) + ".jpg")

align(path_shapes, path_images, path_shapes_norm, test_img_name)
match_average_shape(path_shapes_norm, test_img_path)