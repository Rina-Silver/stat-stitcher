import cv2
import numpy as np
import os
import math

from .feature_detection import *
from .blendings import Blenders, BlenderObj
from .img_trim import trim_all, img_crop
from .color_corrections import color_transfer, Corrections

# класс с вариантами преобразований
class StitchMethods(object):
    HOMOGRAPHY = 'homography'
    AFFINE = 'affine'

# класс с вариантами методов гомографии
class HomographyMethod(object):
    RANSAC = 'RANSAC'
    LMEDS = 'LMEDS'
    RHO = 'RHO'
    REGULAR = 'REGULAR'

HOMO = {
    'RANSAC': cv2.RANSAC,
    'LMEDS': cv2.LMEDS,
    'RHO': cv2.RHO,
    'REGULAR': 0,
}

# класс с настройками сшивки
class StitchParams(object):
    RATIO = 0.75
    MINDIST = 40
    DETECT_OPTIMIZE = True
    OPTIMIZE_BY_LAST = False
    HOMOGRAPHY_FUNC = cv2.RANSAC
    MODE = StitchMethods.HOMOGRAPHY
    SAVE_PER_STEP = False
    CYL_WARP = False
    CYL_F = 1000
    CORR_COL = Corrections.NO
    CORR_CHAIN = False
    NP_TYPE = np.float32

class RuntimeParams(object):
    LAST_WARPED_H = None
    LAST_WARPED_W = None
    LEFT_WARPED_H = None
    LEFT_WARPED_W = None
    RIGHT_WARPED_H = None
    RIGHT_WARPED_W = None

def stitch(query_image, train_image, detectorObj, matcherObj, blenderObj, query_crop=None):
    """Функция применяет сшивку в левом направлении"""
    # расширяем изображение
    query_image = cv2.copyMakeBorder(
        query_image,
        int(query_image.shape[0] * 0.3), 0,
        int(query_image.shape[1] * 0.3), 0,
        cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    if StitchParams.DETECT_OPTIMIZE:
        q = img_crop(query_image, query_crop)
        kp1, des1 = detectorObj.detectAndCompute(q, None)
    else:
        kp1, des1 = detectorObj.detectAndCompute(query_image, None)
    kp2, des2 = detectorObj.detectAndCompute(train_image, None)
    # сопоставление особенностей
    good = matchKeypoints(des1, des2, matcherObj, StitchParams.RATIO, StitchParams.MINDIST)
    if len(good) < 4:
        print(f'Недостаточно хороших точек! Сшивка невозможна')
        return query_image
    # сортировка хороших совпадений
    good = sorted(good, key=lambda x: x.distance)
    if len(good) > 55:
        good = good[:55]
    # Оценка гомографии
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if StitchParams.MODE == StitchMethods.HOMOGRAPHY:
        H, masked = cv2.findHomography(src, dst, StitchParams.HOMOGRAPHY_FUNC, 5.0)
        width = train_image.shape[1] + query_image.shape[1]
        height = train_image.shape[0] + query_image.shape[0]
        result = cv2.warpPerspective(train_image, H, (width, height))
    elif StitchParams.MODE == StitchMethods.AFFINE:
        H, masked = cv2.estimateAffine2D(src, dst, StitchParams.HOMOGRAPHY_FUNC, ransacReprojThreshold=5.0)
        width = train_image.shape[1] + query_image.shape[1]
        height = train_image.shape[0] + query_image.shape[0]
        result = cv2.warpAffine(train_image, H, (width, height))

    # получаем размеры трансформированного изображения
    result_copy = trim_all(result.copy())
    RuntimeParams.LAST_WARPED_H, RuntimeParams.LAST_WARPED_W = result_copy.shape[:2]

    # совмещение и смешение изображений
    a = np.zeros_like(result)
    a[0:query_image.shape[0], 0:query_image.shape[1]] = query_image
    blended = blenderObj.blend(result, a)

    return blended.astype(StitchParams.NP_TYPE)

def right_stitch(query_image, train_image, detectorObj, matcherObj, blenderObj, query_crop=None):
    """Функция применяет сшивку в правом направлении"""
    train_image, query_image = cv2.flip(train_image, +1), cv2.flip(query_image, +1)
    result = cv2.flip(stitch(query_image, train_image, detectorObj, matcherObj, blenderObj, query_crop), +1)
    return result

def cylindricalWarp(img, K):
    """Функция применяет цилиндрическое искажение к изображению и матрице K"""
    h_,w_ = img.shape[:2]
    # координаты пикселей
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # в гомографию
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # нормализованные координаты
    # вычислить цилиндрические координаты
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # проецирование обратно на плоскость
    # возврат от гомогр. координат
    B = B[:,:-1] / B[:,[-1]]
    # обеспечить координаты внутри границ изображения
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    # применяется искажение, возвращается результат
    return cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def cilWarpAuto(img):
    h,w = img.shape[:2]
    f = StitchParams.CYL_F
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    return cylindricalWarp(img, K)

class CylinderPanorama():
    def __init__(self, path, detectorObj, matcherObj, blenderObj):
        filepaths = [os.path.join(path, i) for i in os.listdir(path)]
        self.imagesSorted = []
        for path in filepaths:
            self.imagesSorted.append(cv2.imread(path))

        # присваиваем переданные детектор и сопоставитель полям объекта
        self.detectorObj = detectorObj
        self.matcherObj = matcherObj
        self.blenderObj = blenderObj

    def createPanorama(self):
        # pick the middle image
        cnt = len(self.imagesSorted)
        mid = cnt // 2
        lastInd = cnt - 1
        result = self.imagesSorted[mid]
        if StitchParams.CYL_WARP:
            result = cilWarpAuto(result)
        
        # цикл для поочерёдной (один слева, потом один справа) сшивки
        idx = 1
        leftFlag = True
        leftIter = mid
        rightIter = mid
        while idx < cnt:
            procInd = None
            prevInd = None
            if leftFlag:
                prevInd = leftIter
                leftIter += 1
                procInd = leftIter
            else:
                prevInd = rightIter
                rightIter -= 1
                procInd = rightIter
            
            if procInd > lastInd or procInd < 0:
                leftFlag = not leftFlag
                print(f'Сторона завершена!')
                continue

            print(f'Начат {idx}!')
            train_image = self.imagesSorted[procInd].copy()

            if StitchParams.CORR_COL == Corrections.SFT:
                train_image = color_transfer(self.imagesSorted[prevInd], train_image)
                if StitchParams.CORR_CHAIN:
                    self.imagesSorted[procInd] = train_image

            if StitchParams.CYL_WARP:
                train_image = cilWarpAuto(train_image)
            
            if leftFlag:
                result = stitch(result,train_image,self.detectorObj,self.matcherObj,self.blenderObj, RuntimeParams.LEFT_WARPED_W)
                if StitchParams.OPTIMIZE_BY_LAST:
                    RuntimeParams.LEFT_WARPED_H = RuntimeParams.LAST_WARPED_H
                    RuntimeParams.LEFT_WARPED_W = RuntimeParams.LAST_WARPED_W
            else:
                result = right_stitch(result,train_image,self.detectorObj,self.matcherObj,self.blenderObj, RuntimeParams.RIGHT_WARPED_W)
                if StitchParams.OPTIMIZE_BY_LAST:
                    RuntimeParams.RIGHT_WARPED_H = RuntimeParams.LAST_WARPED_H
                    RuntimeParams.RIGHT_WARPED_W = RuntimeParams.LAST_WARPED_W

            result = trim_all(result)
            leftFlag = not leftFlag

            if StitchParams.SAVE_PER_STEP:
                cv2.imwrite(f'step-st-{idx}.png', result)
                print(f'Сохранён {idx}!')
            print(f'Закончен {idx}!')
            idx += 1
        
        print('Сшивка завершена!')
        return result
