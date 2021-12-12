import numpy as np
import cv2
import sys
import time

    #return lists of DMatch object
    #DMatch.distance : 디스크립터 사이의 거리
    #DMatch.trainIdx : train 디스크립터 속의 디스크립터의 인덱스
    #DMatch.queryIdx : query 디스크립터 속의 디스크립터의 인덱스
    #DMatch.imgIdx : train 이미지의 인덱스


 
############################ SIFT ######################################   
def SIFT(src1, src2):
    
    start = time.time()
    
    state = 1
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints1, desc1 = sift.detectAndCompute(src1, None)
    keypoints2, desc2 = sift.detectAndCompute(src2, None)
    
    print("SIFT processing time: {:.5f}".format(time.time() - start))
    
    return state, keypoints1, keypoints2, desc1, desc2
 
############################ SURF ######################################        
def SURF(src1, src2):
    
    start = time.time()
    state = 1
    surf = cv2.xfeatures2d.SURF_create()
    
    keypoints1, desc1 = surf.detectAndCompute(src1, None)
    keypoints2, desc2 = surf.detectAndCompute(src2, None)
    
    print("SURF processing time: {:.5f}".format(time.time() - start))
    return state, keypoints1, keypoints2, desc1, desc2

############################ ORB ###################################### 
def ORB(src1, src2):
    
    start = time.time()
    
    state=0
    orb = cv2.ORB_create()
    keypoints1, desc1 = orb.detectAndCompute(src1, None)
    keypoints2, desc2 = orb.detectAndCompute(src2, None)
    print("ORB processing time: {:.5f}".format(time.time() - start))
    return state, keypoints1, keypoints2, desc1, desc2

############################ KAZE ######################################
def KAZE(src1, src2):
    
    start = time.time()
    state = 1
    kaze = cv2.KAZE_create()
    keypoints1, desc1 = kaze.detectAndCompute(src1, None)
    keypoints2, desc2 = kaze.detectAndCompute(src2, None)
    
    print("KAZE processing time: {:.5f}".format(time.time() - start))
    return state, keypoints1, keypoints2, desc1, desc2


############################ BF_Match ######################################
def bf_match(visualfeature, method=0):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    if state==1: # SIFT, SURF, KAZE
        matcher = cv2.BFMatcher(cv2.NORM_L2) # L2 Norm 사용
    else: # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # Hamming distance 사용

    BF_match = sorted(matcher.match(desc1, desc2), key = lambda x: x.distance)[:50]
    
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in BF_match]).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in BF_match]).reshape(-1,1,2).astype(np.float32)
    
    H, M = cv2.findHomography(pts1, pts2, method= method)

    if method == 0:
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, BF_match, None, flags=2) 
        
    else: 
        matchMask = M.ravel().tolist()
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, BF_match, None, matchesMask = matchMask, flags=2)
         
    return dst, pts1, pts2

############################ BF_knn ######################################
def bf_knn(visualfeature):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    if state==1: # SIFT, SURF, KAZE
        matcher = cv2.BFMatcher(cv2.NORM_L2) # L2 Norm 사용
    else: # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # Hamming distance 사용
    
    BF_knn = matcher.knnMatch(desc1, desc2, k = 2)
    
    good = []
    pts1 = []
    pts2 = []
    for m,n in BF_knn:
        if m.distance < 0.75*n.distance:          
            good.append([m])
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.trainIdx].pt)
    
    pts1 = np.array(pts1).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array(pts2).reshape(-1,1,2).astype(np.float32)
    
    dst = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, good, None, flags=2)
     
    return dst, pts1, pts2    


############################ flann_match ######################################
def flann_match(visualfeature, method=0):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    
    if state == 1: ## SIFT, SURF, KAZE
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else: # ORB
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number =6, key_size=12, multi_probe_level = 1)
   
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)   
    flann_match = sorted(flann.match(desc1, desc2), key = lambda x: x.distance)[:50]
  
    dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, flann_match, None,
                         flags=2)    

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in flann_match]).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in flann_match]).reshape(-1,1,2).astype(np.float32)
    
    H, M = cv2.findHomography(pts1, pts2, method= method)
    
    if method == 0:
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, flann_match, None, flags=2) 
        
    else: 
        matchMask = M.ravel().tolist()
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, flann_match, None, matchesMask = matchMask, flags=2)
    
    return dst, pts1, pts2  


############################ flann_knn ######################################
def flann_knn(visualfeature):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    
    if state == 1: ## SIFT, SURF, KAZE
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else: # ORB
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number =6, key_size=12, multi_probe_level = 1)
   
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)   
    flann_knn = flann.knnMatch(desc1,desc2,k=2)
    
    matchesMask = [[0,0] for i in range(len(flann_knn))]
    pts1=[]
    pts2=[]
    
    for i,k in enumerate(flann_knn):
        if len(k) == 2:
            if k[0].distance < 0.75*k[1].distance:
                matchesMask[i]=[1,0]
                pts1.append(keypoints1[k[0].queryIdx].pt)
                pts2.append(keypoints2[k[1].trainIdx].pt)

    draw_params = dict(matchColor = None,
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)    
           
    pts1 = np.array(pts1).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array(pts2).reshape(-1,1,2).astype(np.float32)
    dst = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, flann_knn, None, **draw_params)
    
    return dst, pts1, pts2

############################ compare visual feature extracting ######################################   
def compare_visualfeature(SIFT, SURF, ORB, KAZE):
   
    dst1, _, _ = SIFT
    dst2, _, _ = SURF
    dst3, _, _ = ORB
    dst4, _, _ = KAZE
    
    cv2.imshow('SIFT', dst1)
    cv2.imshow('SURF', dst2)
    cv2.imshow('ORB', dst3)
    cv2.imshow('KAZE', dst4)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return

############################ compare matching algorithm ######################################
def compare_matching(BF_match, BF_knn, Flann_match, Flann_knn):
    
    dst1, _, _ = BF_match
    dst2, _, _ = BF_knn
    dst3, _, _ = Flann_match
    dst4, _, _ = Flann_knn
    
    cv2.imshow('BF_match', dst1)
    cv2.imshow('BF_knn', dst2)
    cv2.imshow('Flann_match', dst3)
    cv2.imshow('Flann_knn', dst4)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
           
    return

############################ compare Approximation calculation algorithm ######################################
def compare_method():
    method = [0, cv2.RANSAC, cv2.LMEDS]
    
    dst=[]
    for m in method:
        d, _, _ = bf_match(SIFT(src1, src2), method = m)
        dst.append(d)
        
    cv2.imshow('method = 0', dst[0])
    cv2.imshow('method = RANSAC', dst[1])
    cv2.imshow('method = LMEDS', dst[1])
    
    cv2.waitKey()
    cv2.destroyAllWindows()
                
############################ Calculate Essential Matrix ######################################     
def EssentialMatrix(matching):
    _, pts1, pts2 = matching
    
    e, _ = cv2.findEssentialMat(pts1,pts2, intrinsic_matrix)
    print("BF_match Essential Matrix :\n {}". format(e))    


if __name__ == "__main__":
    
    '''
    python==3.6.0
    opencv-contrib-python==3.3.0.10
    opencv-python==3.3.0.10
    
    '''
    #method = RANSAC, LMEDS
       
    intrinsic_matrix = np.array([  [506.49469633, 0., 236.84841786],\
                                   [0., 675.16797865, 300.61685296],\
                                   [0., 0., 1.]]).astype(np.float32)
             
    # src1 = cv2.resize(cv2.imread('school1.jpg', cv2.IMREAD_GRAYSCALE), dsize=(480,480), interpolation = cv2.INTER_AREA)
    # src2 = cv2.resize(cv2.imread('school2.jpg', cv2.IMREAD_GRAYSCALE), dsize=(480,480), interpolation = cv2.INTER_AREA)
    src1 = cv2.imread('myimg2.jpg', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread('myimg1.jpg', cv2.IMREAD_GRAYSCALE)    
    
    if src1 is None or src2 is None:
        print("Image load failed!")
        sys.exit()
    
    #compare_visualfeature(bf_knn(SIFT(src1, src2)), bf_knn(SURF(src1, src2)), bf_knn(ORB(src1, src2)), bf_knn(KAZE(src1, src2)) )
    #compare_matching(bf_match(SIFT(src1, src2)), bf_knn(SIFT(src1, src2)), flann_match(SIFT(src1, src2)), flann_knn(SIFT(src1, src2))) 
    #compare_method()   
    #EssentialMatrix(bf_knn(ORB(src1, src2)))
