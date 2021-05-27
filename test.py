import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import sys


WIDTH = 400
HEIGHT = 300
""" params ↓"""
EPSILON = 1.0
THRESHOLD = 30


# 이미지를 로드
img = cv2.imread("img/1.png", 1)
img = cv2.resize(img, (WIDTH, HEIGHT))
height, width = img.shape[0:2]

# 이미지 K를 (r, g, b, x, y)의 D로 바꿔준다.
features = np.zeros((height, width, 5))
for y in range(0, height):
    for x in range(0, width):
        arr = img[y][x]
        features[y][x][0] = img[y][x][0]
        features[y][x][1] = img[y][x][1]
        features[y][x][2] = img[y][x][2]
        features[y][x][3] = y
        features[y][x][4] = x

# R 배열을 채워주자.
result = np.zeros((height, width, 3), dtype=np.uint8)
check = np.full((height, width), 1)
current_mode = np.zeros(5)
done = True
data = []
for y in range(height):
    for x in range(width):
        if done:
            current_mode = features[y][x]
            print(f'Y0 = {current_mode}')

        # 나와 같은 무리일 것 같은 친구들을 먼저 고른다.
        window = []
        for i in range(height):
            for j in range(width):
                #if check[i][j] == 0:
                    #continue
                # 현재 모드와 주변 픽셀 간의 특성거리를 구한다.
                feature_dist = 0
                for f in range(5):
                    feature_dist += ((current_mode[f] - features[i][j][f]) ** 2)
                feature_dist = feature_dist**0.5

                # 쓰레쉬 홀딩을 통과했다면, 배열에 입력해준다.
                if feature_dist < THRESHOLD:
                    window.append([i, j])
                print(f'Data selected = {len(window)}')

        mean_R = 0
        mean_G = 0
        mean_B = 0
        mean_i = 0
        mean_j = 0
        current_mean = 0
        mean_col = 0

        # 윈도우 안에 있는 픽셀들의 r, g, b, x, y를 모두 평균내준다.
        for coord in window:
            i, j = coord
            mean_R += features[i][j][0]
            mean_G += features[i][j][1]
            mean_B += features[i][j][2]
            mean_i += features[i][j][3]
            mean_j += features[i][j][4]

        mean_R = mean_R / len(window)
        mean_G = mean_G / len(window)
        mean_B = mean_B / len(window)
        mean_i = mean_i / len(window)
        mean_j = mean_j / len(window)

        # 현재 모드와 구한 평균 사이의 거리합을 구한다.
        mean_e_distance = ((mean_R - current_mode[0])**2
                           + (mean_G - current_mode[1])**2
                           + (mean_B - current_mode[2])**2
                           + (mean_i - current_mode[3])**2
                           + (mean_j - current_mode[4])**2)
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
            for coord in window:
                i, j = coord
                result[i][j] = np.array([mean_R, mean_G, mean_B])
                # 이제 한 번 사용한 것은 사용하지 말자.
                check[i][j] = 0
            done = True

        # 아직 움직일 수 있다면, 모드를 옮겨주자
        else:
            done = False
            current_mode[0] = mean_R
            current_mode[1] = mean_G
            current_mode[2] = mean_B
            current_mode[3] = mean_i
            current_mode[4] = mean_j

        # if(len(total_array) >= 40000:
            # break
#np.save('mean_shifted', result)
cv2.imshow("originImage", img)
cv2.imshow("finalImage", result)
# cv2.imwrite(f'R{THRESHOLD}-E{EPSILON}.png', R)
cv2.waitKey()
