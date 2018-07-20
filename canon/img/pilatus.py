"""
This module handles tiff images captured by Pilatus CCD
"""

from canon.__imports import *


# noinspection PyMethodMayBeStatic,PyClassHasNoInit
class Pilatus:

    @staticmethod
    def __fill_black_space(image):
        """
        param `image` will be modified
        """
        gap_centers = (202 + 212 * i for i in range(4))
        map(lambda c: Pilatus.__fill_horizontal_stripe(image, c), gap_centers)
        Pilatus.__fill_middle_stripe(image)
        image[193, :] = image[192, :]
        image[194, :] = image[193, :]
        image[213, :] = image[214, :]
        image[212, :] = image[213, :]
        image[405, :] = image[404, :]
        image[406, :] = image[405, :]
        image[425, :] = image[426, :]
        image[424, :] = image[425, :]
        image[617, :] = image[616, :]
        image[618, :] = image[617, :]
        image[637, :] = image[638, :]
        image[636, :] = image[637, :]

    @staticmethod
    def __fill_horizontal_stripe(image, xcenter):
        """
        param `image` will be modified
        """
        bottom, top = Pilatus.__find_horizontal_black_edges(image, xcenter)

        if top == bottom: top += 1

        gap = 1. * (top - bottom)
        slope = (image[top, :] - image[bottom, :]) / gap
        intersect = image[top, :] - slope * top

        # TODO:song - there must be a way to vectorize this step
        for x in range(bottom, top + 1):
            image[x, :] = slope * (x - 1) + intersect

    @staticmethod
    def __fill_middle_stripe(image):
        """
        param `image` will be modified.

        This should happen after filling horizontal stripes
        """
        left, right = Pilatus.__find_vertical_black_edges(image, 490)

        if left == right: right += 1

        gap = 1. * (right - left)
        slope = (image[:, right] - image[:, left]) / gap
        intersect = image[:, right] - slope * right

        # TODO:song - there must be a way to vectorize this step
        for y in range(left, right + 1):
            image[:, y] = slope * (y - 1) + intersect

    @staticmethod
    def __find_vertical_black_edges(image, ycenter):
        right = left = ycenter
        right_edge = image[:, right]
        left_edge = image[:, left]

        # auto find edges
        max_iter = 10
        iteration = 0
        while np.argmax(left_edge) <= 0 and iteration < max_iter:
            iteration += 1
            left -= 1
            left_edge = image[:, left]

        iteration = 0
        while np.argmax(right_edge) <= 0 and iteration < max_iter:
            iteration += 1
            right += 1
            right_edge = image[:, right]

        return left, right

    @staticmethod
    def __find_horizontal_black_edges(image, xcenter):
        top = bottom = xcenter
        top_edge = image[top, :]
        bottome_edge = image[bottom, :]

        # auto find edges
        max_iter = 20
        iteration = 0
        while np.argmax(bottome_edge) <= 0 and iteration < max_iter:
            iteration += 1
            bottom -= 1
            bottome_edge = image[bottom, :]

        iteration = 0
        while np.argmax(top_edge) <= 0 and iteration < max_iter:
            iteration += 1
            top += 1
            top_edge = image[top, :]

        return bottom, top

    @staticmethod
    def fill_black(image):
        Pilatus.__fill_black_space(image)
