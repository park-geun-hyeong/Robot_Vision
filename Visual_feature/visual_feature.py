import numpy as np
import cv2
import sys
import time

def good_match(state, keypoints1, keypoints2, desc1, desc2):
    
    ## BFmatcher - Match
    if state==1:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    bf_start = time.time()
    BF_match = sorted(matcher.match(desc1, desc2), key = lambda x: x.distance)[:30]
    
    #return lists of DMatch object
    #DMatch.distance : 디스크립터 사이의 거리
    #DMatch.trainIdx : train 디스크립터 속의 디스크립터의 인덱스
    #DMatch.queryIdx : query 디스크립터 속의 디스크립터의 인덱스
    #DMatch.imgIdx : train 이미지의 인덱스
    
    bf_time = time.time() - bf_start
    
    dst1 = cv2.drawMatches(src1, keypoints1, src2, keypoints2, BF_match, None,
                         flags=2)
    
    ## BFmatcher - KNN Match
    BF_knn = matcher.knnMatch(desc1, desc2, k = 2)
    
    good = []
    for m,n in BF_knn:
        if m.distance < 0.75*n.distance:
            good.append([m])

    dst2 = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, good, None,
                         flags=2)
    
    
    ## FlannBasedMatcher - Match
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    
    if state == 1: ## SIFT, SURF
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else: # ORB
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number =6, key_size=12, multi_probe_level = 1)
   
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    flann_start = time.time()
    flann_match = sorted(flann.match(desc1, desc2), key = lambda x: x.distance)[:30]
    flann_time = time.time() - flann_start
    
    dst3 = cv2.drawMatches(src1, keypoints1, src2, keypoints2, BF_match, None,
                         flags=2)
    
    ##FlannBasedMatcher - Knn Match
    flann_knn = flann.knnMatch(desc1,desc2,k=2)

    matchesMask = [[0,0] for i in range(len(flann_knn))]


    for i,(m,n) in enumerate(flann_knn):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,0,255),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    
    dst4 = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, flann_knn, None, **draw_params)
        
    return dst1, dst2, dst3, dst4, bf_time, flann_time

def SIFT(src1, src2):
    
    start = time.time()
    
    state = 1
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints1, desc1 = sift.detectAndCompute(src1, None)
    keypoints2, desc2 = sift.detectAndCompute(src2, None)
    
    BF_Match, BF_Knn, Flann_Match, Flann_Knn, bf_time, flann_time = \
            good_match(state, keypoints1, keypoints2, desc1, desc2)
    
    print('algorithm processing time: {:.4f} sec'.format(time.time() - start))
    
    cv2.imshow('BF_match, {:.6f} sec'.format(bf_time), BF_Match)
    cv2.imshow('BF_knnmatch', BF_Knn)   
    cv2.imshow('flann_match, {:.6f} sec'.format(flann_time), Flann_Match)
    cv2.imshow('flann_knnmatch', Flann_Knn)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

    return None

def SURF(src1, src2):
    
    start = time.time()
    state = 1
    surf = cv2.xfeatures2d.SURF_create()
    
    keypoints1, desc1 = surf.detectAndCompute(src1, None)
    keypoints2, desc2 = surf.detectAndCompute(src2, None)
    
    BF_Match, BF_Knn, Flann_Match, Flann_Knn, bf_time, flann_time = \
            good_match(state, keypoints1, keypoints2, desc1, desc2)
    
    print('algorithm processing time: {:.4f} sec'.format(time.time() - start))
    
    cv2.imshow('BF_match, {:.6f} sec'.format(bf_time), BF_Match)
    cv2.imshow('BF_knnmatch', BF_Knn)   
    cv2.imshow('flann_match, {:.6f} sec'.format(flann_time), Flann_Match)
    cv2.imshow('flann_knnmatch', Flann_Knn)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

    return None

def ORB(src1, src2):
    
    start = time.time()
    
    state=0
    orb = cv2.ORB_create()
    keypoints1, desc1 = orb.detectAndCompute(src1, None)
    keypoints2, desc2 = orb.detectAndCompute(src2, None)
    
    BF_Match, BF_Knn, Flann_Match, Flann_Knn, bf_time, flann_time = \
            good_match(state, keypoints1, keypoints2, desc1, desc2)
    
    print('algorithm processing time: {:.4f} sec'.format(time.time() - start))
    
    cv2.imshow('BF_match, {:.6f} sec'.format(bf_time), BF_Match)
    cv2.imshow('BF_knnmatch', BF_Knn)   
    cv2.imshow('flann_match, {:.6f} sec'.format(flann_time), Flann_Match)
    cv2.imshow('flann_knnmatch', Flann_Knn)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

    return None

def KAZE(src1, src2):
    
    state = 0 
    
    start = time.time()
    kaze = cv2.KAZE_create()
    keypoints1, desc1 = kaze.detectAndCompute(src1, None)
    keypoints2, desc2 = kaze.detectAndCompute(src2, None)
    
    BF_Match, BF_Knn, Flann_Match, Flann_Knn, bf_time, flann_time = \
            good_match(state, keypoints1, keypoints2, desc1, desc2)
    
    print('algorithm processing time: {:.4f} sec'.format(time.time() - start))
    
    cv2.imshow('BF_match, {:.6f} sec'.format(bf_time), BF_Match)
    cv2.imshow('BF_knnmatch', BF_Knn)   
    cv2.imshow('flann_match, {:.6f} sec'.format(flann_time), Flann_Match)
    cv2.imshow('flann_knnmatch', Flann_Knn)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

    return None

if __name__ == "__main__":
    
    src1 = cv2.imread('./myimg2.jpg', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread('./myimg1.jpg', cv2.IMREAD_GRAYSCALE)
    
    if src1 is None or src2 is None:
        print("Image load failed!")
        sys.exit()
        
        
    #SIFT(src1, src2)
    SURF(src1, src2)
    #ORB(src1, src2)
    #KAZE(src1, src2)
        
    
    
    
