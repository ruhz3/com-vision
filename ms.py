import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import sys


WIDTH = 400
HEIGHT = 300
""" params ↓"""
EPSILON = 5.0
RADIUS = 30


# 이미지를 로드
img = cv2.imread("img/1.png", 1)
img = cv2.resize(img, (WIDTH, HEIGHT))
height, width = img.shape[0:2]

# 이미지 K를 (r, g, b, x, y)의 D로 바꿔준다.
D = np.zeros((height*width, 5))
arr = np.array((1, 3))
idx = 0
for y in range(0, height):
    for x in range(0, width):
        arr = img[y][x]
        D[idx][0] = img[y][x][0]
        D[idx][1] = img[y][x][1]
        D[idx][2] = img[y][x][2]
        D[idx][3] = y
        D[idx][4] = x
        idx += 1

# R 배열을 채워주자.
R = np.zeros((height, width, 3), dtype=np.uint8)
current_mean_random = True
current_mean_arr = np.zeros((1, 5))
below_threshold_arr = []
while len(D) > 0:
    # 현재 모드를 랜덤으로 택한다.
    if current_mean_random:
        current_mean = random.randint(0, len(D)-1)
        for i in range(0, 5):
            current_mean_arr[0][i] = D[current_mean][i]

    # 이번 차례에 검사할 윈도우 범위를 정한다.
    below_threshold_arr = []
    for i in range(0, len(D)):
        ecl_dist = 0
        color_total_current = 0
        color_total_new = 0

        # 현재 모드와의 픽셀 간의 특성거리를 구한다.
        for j in range(0, 5):
            ecl_dist += ((current_mean_arr[0][j] - D[i][j])**2)
        ecl_dist = ecl_dist**0.5

        # 쓰레쉬 홀딩을 통과했다면, 배열에 입력해준다.
        if ecl_dist < RADIUS:
            below_threshold_arr.append(i)

    mean_R = 0
    mean_G = 0
    mean_B = 0
    mean_i = 0
    mean_j = 0
    current_mean = 0
    mean_col = 0

    # 윈도우 안에 있는 픽셀들의 r, g, b, x, y를 모두 평균내준다.
    for i in range(0, len(below_threshold_arr)):
        mean_R += D[below_threshold_arr[i]][0]
        mean_G += D[below_threshold_arr[i]][1]
        mean_B += D[below_threshold_arr[i]][2]
        mean_i += D[below_threshold_arr[i]][3]
        mean_j += D[below_threshold_arr[i]][4]

    mean_R = mean_R / len(below_threshold_arr)
    mean_G = mean_G / len(below_threshold_arr)
    mean_B = mean_B / len(below_threshold_arr)
    mean_i = mean_i / len(below_threshold_arr)
    mean_j = mean_j / len(below_threshold_arr)

    # 현재 모드와 구한 평균 사이의 거리합을 구한다.
    mean_e_distance = ((mean_R - current_mean_arr[0][0])**2
                       + (mean_G - current_mean_arr[0][1])**2
                       + (mean_B - current_mean_arr[0][2])**2
                       + (mean_i - current_mean_arr[0][3])**2
                       + (mean_j - current_mean_arr[0][4])**2)
    mean_e_distance = mean_e_distance**0.5

    nearest_i = 0
    min_e_dist = 0
    counter_threshold = 0

    # 만약 거리가 1 보다 작다면, 그만 움직이자.
    if mean_e_distance < EPSILON:
        new_arr = np.zeros((1, 3))
        new_arr[0][0] = mean_R
        new_arr[0][1] = mean_G
        new_arr[0][2] = mean_B

        # 다음 mode의 좌표를 최대한 근사시켜 입력하자.
        for i in range(0, len(below_threshold_arr)):
            R[int(D[below_threshold_arr[i]][3])][int(D[below_threshold_arr[i]][4])] = np.array([mean_R, mean_G, mean_B])
            # 이제 한 번 사용한 것은 사용하지 말자.
            D[below_threshold_arr[i]][0] = -1

        current_mean_random = True
        new_D = np.zeros((len(D), 5))
        counter_i = 0

        for i in range(0, len(D)):
            if D[i][0] != -1:
                new_D[counter_i][0] = D[i][0]
                new_D[counter_i][1] = D[i][1]
                new_D[counter_i][2] = D[i][2]
                new_D[counter_i][3] = D[i][3]
                new_D[counter_i][4] = D[i][4]
                counter_i += 1

        D = np.zeros((counter_i, 5))
        counter_i -= 1
        for i in range(0, counter_i):
            D[i][0] = new_D[i][0]
            D[i][1] = new_D[i][1]
            D[i][2] = new_D[i][2]
            D[i][3] = new_D[i][3]
            D[i][4] = new_D[i][4]

    # 아직 움직일 수 있다면, 모드를 옮겨주자
    else:
        current_mean_random = False

        current_mean_arr[0][0] = mean_R
        current_mean_arr[0][1] = mean_G
        current_mean_arr[0][2] = mean_B
        current_mean_arr[0][3] = mean_i
        current_mean_arr[0][4] = mean_j

    # if(len(total_array) >= 40000:
        # break
np.save('mean_shifted', R)
cv2.imshow("originImage", img)
cv2.imshow("finalImage", R)
cv2.imwrite(f'R{RADIUS}-E{EPSILON}.png', R)
cv2.waitKey()
