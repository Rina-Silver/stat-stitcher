import numpy as np
import cv2
from .img_trim import  trim_left, trim_right

def preprocess(img1, img2, overlap_w, flag_half):
    if img1.shape[0] != img2.shape[0]:
        print("error: image dimension error")
        exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print("error: overlapped area too large")
        exit()

    w1 = img1.shape[1]
    w2 = img2.shape[1]

    if flag_half:
        shape = np.array(img1.shape)
        shape[1] = int(w1 / 2) + int(w2 / 2)

        subA = np.zeros(shape)
        subA[:, :int(w1 / 2) + int(overlap_w / 2)] = img1[:, :int(w1 / 2) + int(overlap_w / 2)]
        subB = np.zeros(shape)
        subB[:, int(w1 / 2) - int(overlap_w / 2):] = img2[:, w2 - (int(w2 / 2) + int(overlap_w / 2)):]
        mask = np.zeros(shape)
        mask[:, :int(w1 / 2)] = 1
    else:
        shape = np.array(img1.shape)
        shape[1] = w1 + w2 - overlap_w

        subA = np.zeros(shape)
        subA[:, :w1] = img1
        subB = np.zeros(shape)
        subB[:, w1 - overlap_w:] = img2
        mask = np.zeros(shape)
        mask[:, :w1 - int(overlap_w / 2)] = 1

    return subA, subB, mask

def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP

def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.pyrUp(next_img, dstsize=img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP

def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended

def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, dstsize=lev_img.shape[1::-1])
        img += lev_img
    return img

def getBlendMask(im1, im2):
    """
    w, h = im1.shape[:2]
    primBlend = im1.copy()
    primBlend[0:w, 0:h] = im2
    primBlendGray = cv2.cvtColor(primBlend, cv2.COLOR_BGR2GRAY)
    """
    pbl = blend_primitive(im1, im2)
    pbl = cv2.cvtColor(pbl, cv2.COLOR_BGR2GRAY)
    mask = (pbl > 0) * 255
    return mask.astype(np.uint8)

def multiband(img1, img2, overlap_w, leveln=None, flag_half=False):
    if overlap_w < 0:
        print("error: overlap_w should be a positive integer")
        exit()

    subA, subB, mask = preprocess(img1, img2, overlap_w, flag_half)

    fixMask = getBlendMask(subA.astype(np.uint8), subB.astype(np.uint8))

    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print(f"warning: inappropriate number of leveln, max={max_leveln}")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    resultMasked = cv2.bitwise_and(result, result, mask=fixMask)

    return resultMasked

# примитивное смешение (по умолчанию)
def blend_primitive(im1, im2):
    kernel = np.ones((5, 5), np.uint8)
    blended = np.where(cv2.erode(im2, kernel, iterations=1) == 0, im1, im2)
    return blended

# смешение multiband из реализации opencv
def multiband_stitching_api(im1, im2, thresh=10, bands=2):
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(bands)
    h, w = im1.shape[:2]
    blender.prepare((0, 0, w, h)) # called once at start

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    mask1 = (gray1 > thresh) * 255

    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    mask2 = (gray2 > thresh) * 255

    blender.feed(im1.astype(np.int16), mask1.astype(np.uint8), (0,0))
    blender.feed(im2.astype(np.int16), mask2.astype(np.uint8), (0,0))

    res, res_mask = blender.blend(None, None)
    return res.astype(np.uint8)

# смешение feather из реализации opencv
def feather_stitching_api(im1, im2, thresh=10):
    blender = cv2.detail_FeatherBlender()
    blender.setSharpness(0.02)
    h, w = im1.shape[:2]
    blender.prepare((0, 0, w, h)) # called once at start

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    mask1 = gray1 > thresh
    blender.feed(im1.astype(np.int16), mask1.astype(np.uint8), (0,0))

    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    mask2 = gray2 > thresh
    blender.feed(im2.astype(np.int16), mask2.astype(np.uint8), (0,0))

    res, res_mask = blender.blend(None, None)
    return res.astype(np.uint8)

class Blenders(object):
    NO = 'NO'
    FEATHER_API = 'FEATHER_API'
    #MULTIBAND_API = 'MULTIBAND_API'
    MULTIBAND_CUSTOM = 'MULTIBAND_CUSTOM'

# класс, обобщающий остальные
class BlenderObj(object):
    def __init__ (self, blenderStr, bands=8):
        self.bands = bands
        self.type = blenderStr

    def blend(self, im1, im2):
        if self.type == Blenders.NO:
            return blend_primitive(im1, im2)
        elif self.type == Blenders.FEATHER_API:
            return feather_stitching_api(im1, im2)
        #elif self.type == Blenders.MULTIBAND_API:
        #    return multiband_stitching_api(im1, im2, int(bands / 2))
        elif self.type == Blenders.MULTIBAND_CUSTOM:
            width = im1.shape[1]
            im2_crp, im2_right = trim_right(im2)
            im1_crp, im1_left = trim_left(im1)
            overlap = width - im2_right - im1_left
            return multiband(im2_crp, im1_crp, overlap, self.bands)
