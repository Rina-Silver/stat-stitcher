import cv2
import numpy as np

class Detectors(object):
    SIFT = 'sift'
    ORB = 'orb'
    BRISK = 'brisk'
    AKAZE = 'akaze'

class Matchers(object):
    BF = 'bf'
    FLANN_INDEX_LSH = 'flann_lsh'
    FLANN_INDEX_KDTREE = 'flann_kdtree'

class Norms(object):
    NORM_L1 = 'NORM_L1'
    NORM_L2 = 'NORM_L2'
    NORM_HAMMING = 'NORM_HAMMING'
    NORM_HAMMING2 = 'NORM_HAMMING2'

NORM = {
    'NORM_L1': cv2.NORM_L1,
    'NORM_L2': cv2.NORM_L2,
    'NORM_HAMMING': cv2.NORM_HAMMING,
    'NORM_HAMMING2': cv2.NORM_HAMMING2, #If ORB is using VTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used. По умолчанию значение 2, т.е. одновременно выбираются две точки.
}

def getDetector(detectorStr):
    if detectorStr == Detectors.AKAZE:
        return cv2.AKAZE_create()
    if detectorStr == Detectors.BRISK:
        return cv2.BRISK_create()
    if detectorStr == Detectors.ORB:
        return cv2.ORB_create(nfeatures=10000)
    if detectorStr == Detectors.SIFT:
        return cv2.SIFT_create()

class MatcherObj(object):
    def __init__ (self, matcherStr, method=cv2.NORM_HAMMING):
        self.type = matcherStr
        self.norm = method
        if matcherStr == Matchers.BF:
            self.matcher = cv2.BFMatcher(method, crossCheck=False)
        
        if matcherStr == Matchers.FLANN_INDEX_LSH:
            FLANN_INDEX_LSH = 6
            index_params= dict(
                algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1 #2
            )
            search_params = dict(checks=50)   # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        if matcherStr == Matchers.FLANN_INDEX_KDTREE:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

def matchKeypoints(des1, des2, matcherObj, distRatio=0.1, distMin=None):
    if matcherObj.type == Matchers.FLANN_INDEX_KDTREE:
        des1 = np.float32(des1)
        des2 = np.float32(des2)
    knnMatches = matcherObj.matcher.knnMatch(des1, des2, 2)
    good = []
    if distMin is None:
        for row in knnMatches:
            try:
                (m, n) = row
                if m.distance < distRatio * n.distance:
                    good.append(m)
            except:
                print("Недостаточно точек в кортеже!")
    else:
        for row in knnMatches:
            try:
                (m, n) = row
                if m.distance < distRatio * n.distance and m.distance < distMin:
                    good.append(m)
            except:
                print("Недостаточно точек в кортеже!")
    return good

def saveStatFile(pathName, dataDict):
    with open(pathName, 'w', encoding='utf-8') as f:
        for key in dataDict:
            value = dataDict[key]
            f.write(f'{key}:\n{value}\n\n')
        f.close()

