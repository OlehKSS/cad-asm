import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from ashapes import align, match_average_shape
from ashapes.utils import no_points, get_images


# Finds average shape, builds active shapes model
# and compares it with the original shape using hausdorf distance
# Path to hand data
path_shapes = "./data/hand/shapes/shapes.txt"
path_shapes_norm = "./data/hand/shapes/shapes_norm.txt"
path_images = "./data/hand/images"
out_file = "./data/hausdorff.csv"

imgs = get_images(path_images)
imgs = sorted([int(name) for name in imgs])

with open(out_file, 'a+') as file:
    file.write('Image Name,Hausdorff\n')

    for test_img_name in imgs[:20]:
        test_img_path = os.path.join(path_images, str(test_img_name).zfill(4) + ".jpg")

        shapes_norm, test_img_shape = align(path_shapes, path_images, path_shapes_norm, test_img_name)
        res_shape = match_average_shape(shapes_norm, test_img_path, plot_results=False)

        dim_xy = (no_points, 2)
        res_shape_xy = np.zeros(dim_xy)
        res_shape_xy[:, 0] = res_shape[:no_points, 0]
        res_shape_xy[:, 1] = res_shape[no_points:, 0]
        test_img_shape_xy = np.zeros(dim_xy)
        test_img_shape_xy[:, 0] = test_img_shape[:no_points]
        test_img_shape_xy[:, 1] = test_img_shape[no_points:]

        # undirected Hausdorff distance
        h_dist = max(directed_hausdorff(res_shape_xy, test_img_shape_xy)[0],
                                        directed_hausdorff(res_shape_xy, test_img_shape_xy)[0])
        
        file.write(f"{str(test_img_name).zfill(4)},{h_dist}\n")

        img = cv2.imread(test_img_path, 0)
        plt.imshow(img, cmap="gray")
        plt.plot(res_shape_xy[:, 0], res_shape_xy[:, 1])
        plt.plot(test_img_shape_xy[:, 0], test_img_shape_xy[:, 1])
        plt.show()
