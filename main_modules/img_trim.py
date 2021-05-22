import numpy as np

def img_crop(im, width = None):
    k = min(im.shape[1] // 2, im.shape[0] * 2)
    if width is not None:
        k = im.shape[1] - int(width)
    imHalf = im.copy()
    imHalf[:, :k] = 0
    return imHalf

def trim_all(im):
    im, _ = trim_top(im)
    im, _ = trim_bottom(im)
    im, _ = trim_left(im)
    im, _ = trim_right(im)
    return im


def trim_top(im):
    cTop = 0
    while not np.sum(im[0]):
        im = im[1:]
        cTop += 1
    return im, cTop

def trim_bottom(im):
    cBottom = 0
    while not np.sum(im[-1]):
        im = im[:-1]
        cBottom += 1
    return im, cBottom

def trim_left(im):
    cLeft = 0
    while not np.sum(im[:,0]):
        im = im[:,1:]
        cLeft += 1
    return im, cLeft

def trim_right(im):
    cRight = 0
    while not np.sum(im[:,-1]):
        im = im[:,:-1]
        cRight += 1
    return im, cRight
