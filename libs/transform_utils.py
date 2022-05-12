import collections

import cv2
import numpy as np

try:
    import accimage
except ImportError:
    accimage = None


def resize(img, size, interpolation=cv2.INTER_NEAREST):
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    h, w = img.shape[:2]

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation)


def resize_large(img, size, interpolation=cv2.INTER_NEAREST):
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    h, w, _ = img.shape

    if isinstance(size, int):
        if (w >= h and w == size) or (h >= w and h == size):
            return img
        if w > h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation)


def rotatenumpy(image, angle, interpolation=cv2.INTER_NEAREST):
    rot_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=interpolation)
    return result


# good, written with numpy
def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top:top + h, left:left + w] = image
    new_image[:top, left:left + w] = image[top:0:-1, :]
    new_image[top + h:, left:left + w] = image[-1:-bottom - 1:-1, :]
    new_image[:, :left] = new_image[:, left * 2:left:-1]
    new_image[:, left + w:] = new_image[:, -right - 1:-right * 2 - 1:-1]
    return pad_reflection(new_image, next_top, next_bottom,
                          next_left, next_right)


# good, writen with numpy
def pad_constant(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top + h, left:left + w] = image
    return new_image


# change to np/non-np options
def pad_image(mode, image, top, bottom, left, right, value=0):
    if mode == 'reflection':
        return pad_reflection(image, top, bottom, left, right)
    elif mode == 'constant':
        return pad_constant(image, top, bottom, left, right, value)
    else:
        raise ValueError('Unknown mode {}'.format(mode))
