import numpy as np
from math import sqrt, asin, sin, cos
import matplotlib.pyplot as plt

from .utils import no_points, no_shapes


def display_normals(normals, mean, img, scale=15):
    """
    Helper function for displaying normals.

    Input:
        normals (numpy.ndarray): array of normals for each mean.
        mean (numpy.ndarray): array of mean coordinates of landmarks.
        img (numpy.ndarray): object image.
        scale (int): determines how long should normal be on the plot
    """
    x1 = mean[:no_points, 0] + scale * normals[:, 0]
    x2 = mean[:no_points, 0] - scale * normals[:, 0]
    y1 = mean[no_points:, 0] + scale * normals[:, 1]
    y2 = mean[no_points:, 0] - scale * normals[:, 1]

    plt.imshow(img, cmap = "gray")
    plt.plot(mean[:no_points,0], mean[no_points:,0], 'o', color='#F82A80')
    plt.plot((x1, x2), (y1, y2))
    plt.show()


def update_line(hl, new_data):
    """
    Update line on the plot with animation.
    """
    hl.set_xdata(new_data[:no_points])
    hl.set_ydata(new_data[no_points:])
    plt.pause(0.05)
    plt.draw()

class DraggableShape:
    """
    Create draggable shape.

    Paremeters:
        shape: investigated shape.
        shape_plot: plot of the shape to be moved with a rectangle.
    """
    def __init__(self, shape, shape_plot, img):
        self.press = None
        self.rot_press = None
        self.dx = 0
        self.dy = 0
        self.init_shape = shape
        self.shape_x = shape[:no_points]
        self.shape_y = shape[no_points:]
        self.shape_plot = shape_plot
        self.scale = 1

        self.is_transforming = False

        self.update_shape_rect()

        w, h = img.shape
        # coordinates of the origin
        self.img_x0 = int(w / 2)
        self.img_y0 = int(h / 2)

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.shape_plot.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.shape_plot.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.shape_plot.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.scroll_motion = self.shape_plot.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)
        self.key_motion = self.shape_plot.figure.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        # self.key_motion = self.shape_plot.figure.canvas.mpl_connect(
        #     'key_release_event', self.on_key_press)

    def on_press(self, event):
        """On button press we will see if the mouse is over us and store some data"""
        self.update_shape_rect()
        # check whether event is in the area of the shape
        # than we'll do transformation
        if ((self.x0 <= event.xdata <= self.x0 + self.width) and
            (self.y0 <= event.ydata <= self.y0 + self.height)):
            self.press = self.x0, self.y0, event.xdata, event.ydata
        else:
            self.rot_press = self.x0, self.y0, event.xdata, event.ydata

    def on_key_press(self, event):
        """On key press we will move graph in direction of the keys pressed."""
        self.update_shape_rect()

        if event.key == "up":
            dx = 0
            dy = -1
        elif event.key == "down":
            dx = 0
            dy = 1
        elif event.key == "right":
            dx = 1
            dy = 0
        elif event.key == "left":
            dx = -1
            dy = 0

        self.transform_shape(np.eye(2), (dx, dy))

    def on_motion(self, event):
        """On motion we will move the rect if the mouse is over us"""
        if self.press is None and self.rot_press is None: 
            return

        if self.is_transforming:
            return
    
        if ((self.x0 <= event.xdata <= self.x0 + self.width) and
            (self.y0 <= event.ydata <= self.y0 + self.height)):
            # translation
            if self.press is not None:
                self.is_transforming = True
                x0, y0, xpress, ypress = self.press
                self.dx = event.xdata - xpress
                self.dy = event.ydata - ypress
                # decrease movement 10 times to make less sensitive
                self.transform_shape(np.eye(2), (10 * self.dx, 10 * self.dy))
        else:
            # rotation
            if self.rot_press is not None:
                #self.is_transforming = True
                x0, y0, x1, y1 = self.rot_press
                          
                sign = 1 if (event.ydata - y1) >= 0 else -1
                # 0.3 degrees
                unit_angle = sign * 0.0052
                sin_a = sin(unit_angle)
                cos_a = cos(unit_angle)
                R = np.array(((cos_a, -sin_a), (sin_a, cos_a)))
                self.transform_shape(R)

    def on_release(self, event):
        """On release we reset the press data"""
        # self.trans_x = self.shape_x.min() - self.x0
        # self.trans_y = self.shape_y.min() - self.y0

        self.press = None
        self.rot_press = None
        self.is_transforming = False
    
    def on_scroll(self, event):
        """Scaling on scrolling."""
        # 1% percent increase/decrease
        if self.is_transforming:
            return

        if self.scale <= 0.5 and event.step < 0:
            return
        if self.scale >= 1.6 and event.step > 0:
            return

        scale_new = self.scale + event.step * 0.01
        k = scale_new / self.scale
        self.scale = scale_new
        self.transform_shape(k * np.eye(2))
    
    def disconnect(self):
        """Disconnect all the stored connection ids."""
        self.shape_plot.figure.canvas.mpl_disconnect(self.cidpress)
        self.shape_plot.figure.canvas.mpl_disconnect(self.cidrelease)
        self.shape_plot.figure.canvas.mpl_disconnect(self.cidmotion)
        self.shape_plot.figure.canvas.mpl_disconnect(self.key_motion)
        self.shape_plot.figure.canvas.mpl_disconnect(self.scroll_motion)

    
    def update_shape_rect(self):
        """Update information about the shape."""
        self.x0 = self.shape_x.min()
        self.y0 = self.shape_y.min()
        self.width = self.shape_x.max() - self.x0
        self.height = self.shape_y.max() - self.y0
    
    def transform_shape(self, R, t=(0, 0)):
        """
        Apply affine transformation to a shape.
        :param R: Rotation and scaling matrix
        :param tx: translation vector.
        """
        x = np.zeros_like(self.shape_x)
        y = np.zeros_like(self.shape_y)
        mean_x = self.shape_x.mean()
        mean_y = self.shape_y.mean()

        t = np.array(t).reshape((2, 1))
        for i, xy in enumerate(zip(self.shape_x - mean_x, self.shape_y - mean_y)):
            x[i], y[i] = (R @ np.array(xy)) + t

        x += mean_x
        y += mean_y
        self.shape_plot.set_xdata(x)
        self.shape_plot.set_ydata(y)
        self.shape_plot.figure.canvas.draw()

        self.shape_x = x
        self.shape_y = y


def draggable_hand(img, shape):
    """
    Plot dragable hand.

    Parameters:
        img: background image of a hand.
        shape: x,y shape coordinates.

    Returns:

    """

    fig = plt.figure()
    plt.title("Please use arrow keys to move the hand up and down,\nright and left, scroll to zoom,\ndrag outside shape to rotate")
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = "gray")
    shape_plot,  = ax.plot(shape[:no_points, :], shape[no_points:, :])
    dr = DraggableShape(shape, shape_plot, img)
    dr.connect()

    plt.show()

    return dr
