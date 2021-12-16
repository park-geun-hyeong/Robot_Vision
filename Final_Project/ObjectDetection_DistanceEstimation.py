import numpy as np
import cv2
import sys

realwidth = 1.5  # meter
imagewidth = 140  # pixel


def distance(focallength, realwidth, imagewidth):
    dist = ((realwidth + focallength) / imagewidth)
    return dist

def video(FocalLength, video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame)

    orb = cv2.ORB_create()

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if not vidcap.isOpened():
        print('Video open failed!')
        sys.exit()

    cnt = 0
    success = True

    while success:
        success, image = vidcap.read()
        cnt += 1

        try:
            if cnt == 1:
                first_frame = image
                sub_img = first_frame[300:420,120:260]
                base_kp, base_dec = orb.detectAndCompute(sub_img, None)

            else:
                kp, dec = orb.detectAndCompute(image, None)
                matches = matcher.match(base_dec, dec)

                matches = sorted(matches, key=lambda x: x.distance)[:20]

                src_pts = np.float32([base_kp[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                h, w = sub_img.shape[:2]
                pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
                dst = cv2.perspectiveTransform(pts, mtrx)

                cv2.rectangle(image, (int(dst[0][0][0]), int(dst[0][0][1])), (int(dst[2][0][0]), int(dst[2][0][1])), (0,255,0), thickness=2)
                try:
                    dist = distance(FocalLength, realwidth, abs(int(dst[0][0][0]) - int(dst[2][0][0])))
                except ZeroDivisionError:
                    continue
                #print("cnt: {} , dist: {}".format(cnt, dist))

                txt = "distance: {:.2f} M".format(dist)
                cv2.putText(image, txt, (int(dst[0][0][0]-1), int(dst[0][0][1])-1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)

                #print("cnt: {} ,dst: {}".format(cnt ,dst[0][0]))
                #image = cv2.polylines(image, [np.int32(dst)], True, (0,255,0), 2, cv2.LINE_AA)
                #cv2.circle(image , (np.int32(dst[0][0][0]), np.int32(dst[0][0][1])), 5, (0,0,255), -1)
                #cv2.circle(image, (np.int32(dst[2][0][0]), np.int32(dst[2][0][1])), 5, (255, 0, 0), -1)

                #matchesMask = mask.ravel().tolist()
                #dst = cv2.drawMatches(sub_img, base_kp, image, kp, matches, None, \
                #                       matchesMask = matchesMask,
                #                       flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                if cnt % 2 == 0:
                    cv2.imshow('vid', image)

        except:
            print("cnt: {}, dst: {}".format(cnt, dist))
            continue


        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":

    fx = 506.49469633
    fy = 675.16797865
    focal_length = (fx + fy) / 2

    video(focal_length, "./car.mp4")






