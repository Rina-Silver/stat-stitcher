
from os import path, mkdir
import datetime
import time
import traceback
import cv2
import numpy as np

from main_modules.feature_detection import *
from main_modules.panorama import *

scr_dir = path.dirname(path.abspath(__file__))

# текущее время, берём для создания папки с результатами
now = datetime.datetime.now()
now = now.strftime("%d-%m-%Y %H.%M")
tDir = f'{scr_dir}/{now}'
mkdir(tDir)

img1_path = scr_dir+'/img-article/squaw_peak2_600.jpg'
img2_path = scr_dir+'/img-article/squaw_peak3_600.jpg'
#img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# цикл по всем методам поиска точек
for pDetector in [Detectors.AKAZE, Detectors.BRISK, Detectors.SIFT, Detectors.ORB]:
    print(f"В работе детектор: {pDetector}")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # общая статистика для детектора
    dDict = {}

    # делаем папку для метода
    tDDir = f'{tDir}/{pDetector}'
    mkdir(tDDir)

    # находим ключевые точки
    detector = getDetector(pDetector)
    start = int(round(time.time() * 1000))
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    finish = int(round(time.time() * 1000))
    time_diff = finish - start
    #print(f"Время поиска точек, мс: {time_diff}")
    dDict['Время поиска точек, мс'] = time_diff
    # количества
    cnt1 = len(keypoints1)
    cnt2 = len(keypoints2)
    #print(f"Всего точек, перв. изобр., {cnt1}")
    #print(f"Всего точек, втор. изобр., {cnt2}")
    dDict['Всего точек, перв. изобр.'] = cnt1
    dDict['Всего точек, втор. изобр.'] = cnt2

    # цикл по способам сравнения
    for pMatcher in [Matchers.BF, Matchers.FLANN_INDEX_KDTREE, Matchers.FLANN_INDEX_LSH]:
        # цикл по метрикам (если FLANN, то всего один проход)
        for pNorm in [Norms.NORM_L1, Norms.NORM_L2, Norms.NORM_HAMMING]:
            pMethod = NORM[pNorm]
            print(f"Метод {pMatcher}, метрика {pNorm} ({pMethod})")

            # определяем имя подпапки
            tName = pMatcher
            tDMDir = f'{tDDir}/{pMatcher}'
            # если сейчас BF - то указывам метрику в названии папки
            if pMatcher == Matchers.BF:
                tName = f'{tName}-{pNorm}'
                tDMDir = f'{tDMDir}-{pNorm}'
            
            try:
                # статистика для метода сопоставления и метрики
                dMMDict = {}
                # пытаемся выполнить сопоставление
                start = int(round(time.time() * 1000))
                matcherObj = MatcherObj(pMatcher, pMethod)
                matches = matchKeypoints(
                    descriptors1,
                    descriptors2,
                    matcherObj,
                    0.5
                )
                finish = int(round(time.time() * 1000))
                time_diff = finish - start
                #print(f"Время сравнения точек, мс: {time_diff}")
                dMMDict['Время сравнения точек, мс'] = time_diff

                mkdir(tDMDir)

                # наша базовая статистика
                cnt3 = len(matches)
                prc = float(cnt3/cnt1) * 100
                #print(f"Всего {cnt3} - совпадении, {prc} %")
                dMMDict['Совпадений, шт.'] = cnt3
                dMMDict['Совпадений, %'] = prc

                # сохраняем сравнение всех
                compare_all = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
                cv2.imwrite(f'{tDMDir}/compare_all.jpg', compare_all)
                
                # вычисляем гомографию изображений
                src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

                # пройдёмся по вариантам методов гомографии
                for pHomo in [HomographyMethod.RANSAC, HomographyMethod.LMEDS, HomographyMethod.RHO, HomographyMethod.REGULAR]:
                    pHomoMethod = HOMO[pHomo]
                    # статистика для метода гомографии
                    dMMHDict = {}
                    M, mask = cv2.findHomography(src_pts, dst_pts, pHomoMethod, 5.0)
                    matchesMask = mask.ravel().tolist()
                    inliersCnt = np.sum(mask)
                    #print(f"Правильных (по гомографии) совпадений: {inliersCnt}")
                    dMMHDict['Правильных (по гомографии) совпадений'] = inliersCnt
                    inlier_ratio = (inliersCnt / float(len(matches))) * 100
                    #print(f"Правильных (по гомографии) совпадений, % от совпадений: {inlier_ratio}")
                    dMMHDict['Правильных (по гомографии) совпадений, % от совпадений'] = inlier_ratio

                    # сохраняем сравнение правильных
                    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
                    compare_inliers = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)
                    cv2.imwrite(f'{tDMDir}/compare_inliers_{pHomo}.jpg', compare_inliers)
                    saveStatFile(f'{tDMDir}/homo-{pHomo}.txt', dMMHDict)

                # сохраняем общую статистику по случаю сравнения/метрики для детектора
                saveStatFile(f'{tDMDir}/case-{tName}.txt', dMMDict)

            except Exception as e:
                print(f"Ошибка при {pDetector}, {pMatcher}, {pNorm}")
                print(traceback.format_exc())

            # если сейчас FLANN - то нет смысла проходить по метрикам
            if pMatcher != Matchers.BF:
                break
    
    # сохраняем картинки с точками
    img1dkp = cv2.drawKeypoints(img1, keypoints1, None)
    cv2.imwrite(f'{tDDir}/img1kp.jpg', img1dkp)
    img2dkp = cv2.drawKeypoints(img2, keypoints2, None)
    cv2.imwrite(f'{tDDir}/img2kp.jpg', img2dkp)

    # сохраняем общую статистику по детектору
    saveStatFile(f'{tDDir}/common-{pDetector}.txt', dDict)
