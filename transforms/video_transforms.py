import numbers
import random
import numpy as np
import PIL
import skimage.transform
import torchvision
import math
import torch
from torchvision import transforms

import utils.video_utils as F

# Thanks to : https://github.com/hassony2/torch_videovision


class I3DPixelsValue(object):
    """
    Scale the pixel value between -1 and 1 instead of 0 and 1 (required for I3D)
    """

    def __call__(self, sample):
        return sample * 2 - 1


class ChangeVideoShape(object):
    """
    Expect to receive a ndarray of chape (Time, Height, Width, Channel) which is the default format
    of cv2 or PIL. Change the shape of the ndarray to TCHW or CTHW.
    """

    def __init__(self, shape: str):
        """
        shape : a string with the value "CTHW" or "TCHW".
        """

        self.shape = shape

    def __call__(self, sample):

        if self.shape == "CTHW":
            sample = np.transpose(sample, (3, 0, 1, 2))
        elif self.shape == "TCHW":
            sample = np.transpose(sample, (0, 3, 1, 2))
        else:
            raise ValueError(f"Received {self.shape}. Expecting TCHW or CTHW.")

        return sample


class ResizeVideo(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation="nearest"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(clip, self.size, interpolation=self.interpolation)
        return np.array(resized)


class RandomCropVideo(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return np.array(cropped)


class CenterCropVideo(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        numpy.ndarray: Cropped list of images of shape (t, h, w, c)
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image"
                + "but got list of {0}".format(type(clip[0]))
            )
        if w > im_w or h > im_h:
            error_msg = (
                "Initial image size should be larger then "
                "cropped size but got cropped sizes : ({w}, {h}) while "
                "initial image is ({im_w}, {im_h})".format(
                    im_w=im_w, im_h=im_h, w=w, h=h
                )
            )
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.0))
        y1 = int(round((im_h - h) / 2.0))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return np.array(cropped)


class TrimVideo(object):
    """Trim each video the same way. Waiting shape TCHW
    """

    def __init__(self, size, offset=None):
        self.end = size
        self.begin = 0

        if offset != None:
            self.begin = offset
            self.end += offset

    def __call__(self, clip):
        resized = clip

        if len(clip) > self.end:
            resized = clip[self.begin : self.end]
        return np.array(resized)


class RandomTrimVideo(object):
    """Trim randomly the video. Waiting shape TCHW
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        resized = clip

        if len(clip) > self.size:
            diff = len(resized) - self.size

            start = random.randint(0, diff)
            end = start + self.size

            resized = resized[start:end]

        return np.array(resized)


class PadVideo(object):
    def __init__(self, size, loop=True):
        self.size = size
        self.loop = loop

    def __call__(self, clip):
        if self.loop:
            resized = self._loop_sequence(clip, self.size)
        else:
            resized = self._pad_sequence(clip, self.size)

        return np.array(resized)

    def _pad_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])

        zero_arr = np.zeros(new_shape)
        zero_arr[: shape[0]] = sequence

        return zero_arr

    def _loop_sequence(self, sequence, length):
        shape = sequence.shape
        new_shape = (length, shape[1], shape[2], shape[3])
        zero_arr = np.zeros(new_shape)

        video_len = len(sequence)

        for i in range(length):
            vid_idx = i % video_len
            zero_arr[i] = sequence[vid_idx]

        return zero_arr


class RandomColorJitterVideo(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
 
        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        # Apply to all images
        jittered_clip = []
        for img in clip:
            # Transforming frame
            frame = img * 255
            pillow_frame = transforms.ToPILImage()(np.uint8(frame))

            for func in img_transforms:
                jittered_img = func(pillow_frame)
            jittered_clip.append(np.array(jittered_img))


        return np.array(jittered_clip) / 255


