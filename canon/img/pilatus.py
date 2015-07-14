"""
This module handles tiff images captured by Pilatus CCD
"""

from ..__imports import *


# noinspection PyMethodMayBeStatic
class Pilatus:
    def __init__(self):
        pass

    @staticmethod
    def __remove_background(image):
        """
        param `image` will be modified
        """
        assert isinstance(image, np.ndarray)

        Pilatus.__fill_black_space(image)

    @staticmethod
    def __fill_black_space(image):
        """
        param `image` will be modified
        """
        gap_centers = (202 + 212 * i for i in xrange(4))
        map(lambda c: Pilatus.__fill_horizontal_stripe(image, c), gap_centers)
        Pilatus.__fill_middle_stripe(image)

    @staticmethod
    def __fill_horizontal_stripe(image, xcenter):
        """
        param `image` will be modified
        """
        bottom, top = Pilatus.__find_horizontal_black_edges(image, xcenter)

        if top == bottom: top += 1

        # TODO:song - there must be a way to vectorize this step
        for x in xrange(bottom, top + 1):
            image[x, :] = (image[top + 1, :] + image[bottom - 1, :]) / 2.

    @staticmethod
    def __fill_middle_stripe(image):
        """
        param `image` will be modified.

        This should happen after filling horizontal stripes
        """
        left, right = Pilatus.__find_vertical_black_edges(image, 490)

        if left == right: right += 1

        # TODO:song - there must be a way to vectorize this step
        for y in xrange(left, right + 1):
            image[:, y] = (image[:, left - 1] + image[:, right + 1]) / 2.

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
    def remove_background(image):
        Pilatus.__remove_background(image)

    @staticmethod
    def fill_black(image):
        Pilatus.__fill_black_space(image)
