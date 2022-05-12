import collections
import numbers
import random

import cv2
import torch

from libs.transform_utils import rotatenumpy, pad_image, resize


class Scale(object):
    """docstring for Scale"""

    def __init__(self, short_side):
        self.short_side = short_side

    def __call__(self, pair_0, pair_1):
        if (type(self.short_side) == int):
            h, w, c = pair_0.shape
            if (h > w):
                tw = self.short_side
                th = (tw * h) / w
                th = int((th // 64) * 64)
            else:
                th = self.short_side
                tw = (th * w) / h
                tw = int((tw // 64) * 64)
        elif (type(self.short_side) == list):
            th = self.short_side[0]
            tw = self.short_side[1]

        interpolation = cv2.INTER_NEAREST
        pair_0 = cv2.resize(pair_0, dsize=(tw, th), interpolation=interpolation)
        pair_1 = cv2.resize(pair_1, dsize=(tw, th), interpolation=interpolation)

        return pair_0, pair_1


class RandomCrop(object):
    """for pair of frames"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, pair_0, pair_1):
        h, w, c = pair_0.shape
        th, tw = self.size
        top = bottom = left = right = 0

        if w == tw and h == th:
            return pair_0, pair_1

        if w < tw:
            left = (tw - w) // 2
            right = tw - w - left
        if h < th:
            top = (th - h) // 2
            bottom = th - h - top
        if left > 0 or right > 0 or top > 0 or bottom > 0:
            pair_0 = pad_image(
                'reflection', pair_0, top, bottom, left, right)
            pair_1 = pad_image(
                'reflection', pair_1, top, bottom, left, right)

        if w > tw:
            x1 = random.randint(0, w - tw)
            pair_0 = pair_0[:, x1:x1 + tw]
            pair_1 = pair_1[:, x1:x1 + tw]
        if h > th:
            y1 = random.randint(0, h - th)
            pair_0 = pair_0[y1:y1 + th]
            pair_1 = pair_1[y1:y1 + th]

        return pair_0, pair_1


class CenterCrop(object):
    """for pair of frames"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *frames):
        frames = list(frames)
        h, w, c = frames[0].shape
        th, tw = self.size
        top = bottom = left = right = 0

        if w == tw and h == th:
            return frames

        if w < tw:
            left = (tw - w) // 2
            right = tw - w - left
        if h < th:
            top = (th - h) // 2
            bottom = th - h - top
        if left > 0 or right > 0 or top > 0 or bottom > 0:
            for i in range(len(frames)):
                frames[i] = pad_image(
                    'reflection', frames[i], top, bottom, left, right)

        if w > tw:
            # x1 = random.randint(0, w - tw)
            x1 = (w - tw) // 2
            for i in range(len(frames)):
                frames[i] = frames[i][:, x1:x1 + tw]
        if h > th:
            # y1 = random.randint(0, h - th)
            y1 = (h - th) // 2
            for i in range(len(frames)):
                frames[i] = frames[i][y1:y1 + th]

        return frames


class RandomScale(object):
    """docstring for RandomScale"""

    def __init__(self, scale):
        if isinstance(scale, numbers.Number):
            scale = [1 / scale, scale]
        self.scale = scale

    def __call__(self, pair_0, pair_1):
        ratio = random.uniform(self.scale[0], self.scale[1])
        h, w, c = pair_0.shape
        tw = int(ratio * w)
        th = int(ratio * h)
        if ratio == 1:
            return pair_0, pair_1
        elif ratio < 1:
            interpolation = cv2.INTER_LANCZOS4
        elif ratio > 1:
            interpolation = cv2.INTER_CUBIC
        pair_0 = cv2.resize(pair_0, dsize=(tw, th), interpolation=interpolation)
        pair_1 = cv2.resize(pair_1, dsize=(tw, th), interpolation=interpolation)

        return pair_0, pair_1


class RandomRotate(object):
    """docstring for RandomRotate"""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, pair_0, pair_1):
        h, w, c = pair_0.shape
        p = max((h, w))
        angle = random.randint(0, self.angle * 2) - self.angle
        pair_0 = pad_image('reflection', pair_0, h, h, w, w)
        pair_0 = rotatenumpy(pair_0, angle)
        pair_0 = pair_0[h: h + h, w: w + w]
        pair_1 = pad_image('reflection', pair_1, h, h, w, w)
        pair_1 = rotatenumpy(pair_1, angle)
        pair_1 = pair_1[h: h + h, w: w + w]

        return pair_0, pair_1


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, pair_0, pair_1):
        if random.random() < 0.5:
            results = [cv2.flip(pair_0, 1), cv2.flip(pair_1, 1)]
        else:
            results = [pair_0, pair_1]
        return results


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pair_0, pair_1):
        return resize(pair_0, self.size, self.interpolation), \
               resize(pair_1, self.size, self.interpolation)


class Pad(object):

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, pair_0, pair_1):
        if self.fill == -1:
            pair_0 = pad_image('reflection', pair_0,
                               self.padding, self.padding, self.padding, self.padding)
            pair_1 = pad_image('reflection', pair_1,
                               self.padding, self.padding, self.padding, self.padding)
        else:
            pair_0 = pad_image('constant', pair_0,
                               self.padding, self.padding, self.padding, self.padding,
                               value=self.fill)
            pair_1 = pad_image('constant', pair_1,
                               self.padding, self.padding, self.padding, self.padding,
                               value=self.fill)

        return pair_0, pair_1


class ResizeandPad(object):
    """
    resize the larger boundary to the desized eva_size;
    pad the smaller one to square
    """

    def __init__(self, size, interpolation=cv2.INTER_NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pair_0, pair_1):
        """
        Resize and Pad
        """
        pair_0 = resize(pair_0, self.size, self.interpolation)
        pair_1 = resize(pair_1, self.size, self.interpolation)
        h, w, _ = pair_0.shape
        if w > h:
            bd = int((w - h) / 2)
            pair_0 = pad_image('reflection', pair_0, bd, (w - h) - bd, 0, 0)
            pair_1 = pad_image('reflection', pair_1, bd, (w - h) - bd, 0, 0)
        elif h > w:
            bd = int((h - w) / 2)
            pair_0 = pad_image('reflection', pair_0, 0, 0, bd, (h - w) - bd)
            pair_1 = pad_image('reflection', pair_1, 0, 0, bd, (h - w) - bd)
        return pair_0, pair_1


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std=(1.0, 1.0, 1.0)):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, pair_0, pair_1):
        for t, m, s in zip(pair_0, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(pair_1, self.mean, self.std):
            t.sub_(m).div_(s)
        # print("pair_0: ", pair_0.size())
        # print("pair_1: ", pair_1.size())
        return pair_0, pair_1


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pair_0, pair_1):
        if (pair_0.ndim == 2):
            pair_0 = torch.from_numpy(pair_0.copy()).contiguous().float().unsqueeze(0)
            pair_1 = torch.from_numpy(pair_1.copy()).contiguous().float().unsqueeze(0)
        else:
            pair_0 = torch.from_numpy(pair_0.transpose(2, 0, 1).copy()).contiguous().float()
            pair_1 = torch.from_numpy(pair_1.transpose(2, 0, 1).copy()).contiguous().float()
        return pair_0, pair_1


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
