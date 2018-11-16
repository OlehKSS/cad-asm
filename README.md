# cad-asm
Active Shapes Method for object detection and segmentation

Find two scripts in the examples folder for testing data

1.  find_shape.py finds average shape, builds an active shapes model and tests it
    Provide path_shapes: path to 'shapes.txt' file
            path_images: path to 'images' folder
            test_img_name: number of image you would like to model

2.  find_hausdorff.py Finds average shape, builds active shapes model,
            and compares it with the original shape using hausdorff distance
    Provide path_shapes: path to 'shapes.txt' file
            path_images: path to 'images' folder
            out_file: path to save 'hausdorff.csv' file
            imgs[:20] (line 29) specifies which images (or all) you would like to perform Hausdorff
                calculations for. Here is it from 0-20.