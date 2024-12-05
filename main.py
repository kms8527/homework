import argparse
import cv2
import os
import numpy as np
import copy
import random
import sys

parser = argparse.ArgumentParser(description=' robot vision epipolor homework')
parser.add_argument('--name1', default='Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg', type=str,
                    help='image1 name')
parser.add_argument('--name2', default='Episcopal Gaudi/4386465943_8cf9776378_o.jpg', type=str,
                    help='image2 name')
parser.add_argument('--resize',default=[1072,712], type=list,
                    help = 'resize image shape of image')
parser.add_argument('--get-feature', default=True, type=bool,
                    help='get feature')
parser.add_argument('--dir', default='/home/a/Homework/robotvision/homework2/data/', type=str,
                    help='image dir')
parser.add_argument('--point1_name', default='pts2d-pic_a.txt', type=str,
                    help='image dir')
parser.add_argument('--point2_name', default='pts2d-pic_b.txt', type=str,
                    help='image dir')
parser.add_argument('--point3d_name', default='pts3d.txt', type=str,
                    help='image dir')
parser.add_argument('--SamplingNum', default=8, type=int,
                    help='The number of points to use while estimating the fundamental matrix')
parser.add_argument('--ErrorThreshold', default=0.005, type=float,
                    help='The threshold for the consensus set')
parser.add_argument('--IterNum', default=10000, type=float,
                    help='RANSAC Iteration number')

def get_image(*args):
    """
    :param args:
    :return:
    resize도 여기서 함
    """

    imgA = cv2.imread(args[0].dir + args[0].name1, cv2.IMREAD_COLOR)
    imgB = cv2.imread(args[0].dir + args[0].name2, cv2.IMREAD_COLOR)

    assert imgA is not None, 'image dir error'
    assert imgB is not None, 'image dir error'

    imgA = cv2.resize(imgA, args[0].resize)
    imgB = cv2.resize(imgB, args[0].resize)


    # img = cv2.resize(img, args[0].resize)
    return imgA, imgB


def read_point(*args):
    p = [[], [], []]
    file_names = [args[0].point1_name, args[0].point2_name, args[0].point3d_name]

    for idx, file_name in enumerate(file_names):
        file = open(args[0].dir + file_name, "r")
        strings = file.readlines()
        for string in strings:
            newstr = string.strip().split()
            if idx < 2:
                p[idx].append([float(newstr[0]), float(newstr[1])])
            else:
                p[idx].append([float(newstr[0]), float(newstr[1]), float(newstr[2])])

        file.close()
    return p[0], p[1], p[2]


def GetProjectionMatrix(point2D, point3D):
    n = len(point3D)
    A = np.zeros([2 * n, 12])
    for i in range(n):
        A[2 * i, 0:4] = point3D[i][:] + [1.0]
        A[2 * i, -4:] = list(np.array(point3D[i][:] + [1.0]) * -point2D[i][0])
        A[2 * i + 1, 4:8] = point3D[i][:] + [1.0]
        A[2 * i + 1, -4:] = list(np.array(point3D[i][:] + [1.0]) * -point2D[i][1])

    U, s, V = np.linalg.svd(A, full_matrices=True)
    # S = np.zeros(A.shape)
    #
    # for i in range(len(s)):
    #     S[i][i] = s[i]
    # appA = np.dot(U, np.dot(S, V))
    p1 = V[-1, 0:4]
    p2 = V[-1, 4:8]
    p3 = V[-1, 8:]
    P = np.array([p1, p2, p3])

    return P

def ComputeReprojectionError(P, point2D, point3D):
    p = []
    for i in range(len(point3D)):
        x, y, z = P @ np.transpose(np.array([point3D[i] + [1]]))
        x = float(x / z)
        y = float(y / z)
        p.append([x, y])
        # print(x, y)
    return np.linalg.norm(np.array(point2D) - np.array(p))


def GetFundamentalMatrix(p1, p2):
    assert len(p1) == len(p2), 'p1, p2 포인트 수 항상 같아야 함.'
    M = len(p1)
    A = np.ones([M, 9])
    for i in range(M):
        A[i, :] = [p1[i][0] * p2[i][0], p1[i][0] * p2[i][1], p1[i][0], p1[i][1] * p2[i][0], p1[i][1] * p2[i][1],
                   p1[i][1], p2[i][0], p2[i][1], 1]

    U, s, V = np.linalg.svd(A, full_matrices=True)

    F1 = V[-1, 0:3]  # F의 1행 벡터
    F2 = V[-1, 3:6]
    F3 = V[-1, 6:]
    F = np.array([F1, F2, F3])

    # 검산#
    # for i in range(M):
    #     print(np.array([p2[i]+[1]]) @ F @ np.transpose(np.array([p1[i]+[1]])))
    #   #
    return F


def DrawEpipolarLine(F, points, points2, img):
    epiplar_img = copy.copy(img)
    for idx, p in enumerate(points):
        [a, b, c] = F @ np.transpose(np.array(p + [1]))
        cv2.line(epiplar_img, (img.shape[1], int(-(c + a * img.shape[1]) / b)), (0, int(-c / b)), (0, 0, 255), 1,
                 cv2.LINE_AA)
        tmp = (int(points2[idx][0]),int(points2[idx][1]))
        cv2.line(epiplar_img, tmp, tmp, (255, 255, 255), 5, cv2.LINE_AA)

    cv2.imshow('raw image', img)
    cv2.imshow('epipolar', epiplar_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return epiplar_img


def SiftMatching(imgA, imgB):
    # Input : image1 and image2 in opencv format
    # Output : corresponding keypoints for source and target images
    # Output Format : Numpy matrix of shape: [No. of Correspondences X 2]
    sift = cv2.SIFT_create(10000)
    # surf = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(imgA, None)
    kp2, des2 = sift.detectAndCompute(imgB, None)

    matcher = cv2.BFMatcher_create()
    # matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    matches = matcher.match(des1, des2)
    # 좋은 매칭 결과 선별
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:100]
    # 상위 80개만 선별

    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]
                    ).reshape(-1, 1, 2).astype(np.float32)

    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]
                    ).reshape(-1, 1, 2).astype(np.float32)

    #####디버그 #####
    # H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    #
    # dst = cv2.drawMatches(imgA, kp1, imgB, kp2, good_matches, None,
    #                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    # cv2.imshow('dst', dst)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # cv2.imwrite("dst",dst)
    ################

    return pts1.squeeze(), pts2.squeeze(), good_matches


def EstimateFundamentalMatrix(pts1, pts2, SamplingNum, ErrorThreshold, IterNum):
    """
    :param pts1: list([x1, y1], [x2, y2] ...)
    :param pts2: list([x1, y1], [x2, y2] ...)
    RANSAC 사용
    :return:
    """
    maxInliers = 0
    for i in range(IterNum):
        idxs = random.sample(range(1, 100), SamplingNum)
        F = GetFundamentalMatrix(pts1[idxs], pts2[idxs])
        CurrentInliers = ComputeInlierSize(pts1.tolist(), pts2.tolist(), F, ErrorThreshold)
        if CurrentInliers > maxInliers:
            maxInliers = CurrentInliers
            BestF = F
            BestIdxs = idxs

    return BestIdxs, BestF


def ComputeInlierSize(p1, p2, F, ErrorThreshold):
    assert len(p1) == len(p2), 'p1, p2 포인트 수 항상 같아야 함.'
    M = len(p1)
    ErrorThreshold
    InlierSize = 0
    for i in range(M):
        Error = abs(np.array([p2[i] + [1]]) @ F @ np.transpose(np.array([p1[i] + [1]])))
        if Error < ErrorThreshold:
            InlierSize += 1

    return InlierSize


def main():
    args = parser.parse_args()


    if not args.get_feature:
        imgA, imgB = get_image(args)
        p1, p2, p3 = read_point(args)
        P1 = GetProjectionMatrix(p1, p3)
        P2 = GetProjectionMatrix(p2, p3)
        print(P1, P2)
        E1 = ComputeReprojectionError(P1, p1, p3)
        E2 = ComputeReprojectionError(P2, p2, p3)
        print(E1, E2)
        F = GetFundamentalMatrix(p1, p2)
        img1 = DrawEpipolarLine(np.transpose(F), p1, p2, imgB)
        img2 = DrawEpipolarLine(F, p2, p1, imgA)
        cv2.imwrite('img1.jpg', img1)
        cv2.imwrite('img2.jpg', img2)

    else:
        SamplingNum = args.SamplingNum
        ErrorThreshold = args.ErrorThreshold
        IterNum = args.IterNum

        # 3번 문제
        imgA, imgB = get_image(args)
        pts1, pts2, good_matches = SiftMatching(imgA, imgB)
        idxs, F = EstimateFundamentalMatrix(pts1, pts2, SamplingNum, ErrorThreshold, IterNum)
        img1 = DrawEpipolarLine(np.transpose(F), pts1[idxs].tolist(), pts2[idxs].tolist(), imgB)
        img2 = DrawEpipolarLine(F, pts2[idxs].tolist(), pts1[idxs].tolist(), imgA)
        cv2.imwrite("SN_"+str(SamplingNum)+'_Threshold_'+str(ErrorThreshold)+'_IterNum_'+str(IterNum)+"_1.jpg", img1)
        cv2.imwrite("SN_"+str(SamplingNum)+'_Threshold_'+str(ErrorThreshold)+'_IterNum_'+str(IterNum)+"_2.jpg", img2)



if __name__ == '__main__':
    main()
