import cv2
import numpy as np
from math import sqrt
N_SIZE = 3
DERIVATIVE_X = [1, -1]
DERIVATIVE_Y = [[1],
                [-1]]

# 이미지를 불러온다.
color_first_img = cv2.imread("img/11.png", 1)
color_second_img = cv2.imread("img/22.png", 1)
first_img = cv2.imread("img/11.png", 0)
second_img = cv2.imread("img/22.png", 0)
height, width = first_img.shape

# df/dy, df/dx, df/dt를 찾는다.
first_img = cv2.boxFilter(first_img, -1, (5, 5))
second_img = cv2.boxFilter(second_img, -1, (5, 5))
df_dy = cv2.filter2D(first_img, ddepth=cv2.CV_64F, kernel=np.array(DERIVATIVE_Y))
df_dx = cv2.filter2D(first_img, ddepth=cv2.CV_64F, kernel=np.array(DERIVATIVE_X))
df_dt = second_img-first_img

# A, b를 구성하고, v를 구한다.
gap = int(N_SIZE/2)
result = np.empty(shape=(height, width, 2))
for y in range(1, height-1, gap):
    for x in range(1, width-1, gap):
        try:
            # N 범위에 있는 픽셀들을 모아준다.
            y_tmp = df_dy[(y - gap):(y + gap + 1), (x - gap):(x + gap + 1)]
            x_tmp = df_dx[(y - gap):(y + gap + 1), (x - gap):(x + gap + 1)]
            t_tmp = df_dt[(y - gap):(y + gap + 1), (x - gap):(x + gap + 1)]

            # A, v, b 행렬을 구성한다.
            A = cv2.hconcat([np.reshape(y_tmp, (N_SIZE ** 2, 1)), np.reshape(x_tmp, (N_SIZE ** 2, 1))])
            A_T = np.transpose(A)
            b = -np.reshape(t_tmp, (N_SIZE ** 2, 1))
            v_T = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_T, A)), A_T), b)

            # 결과 (v, u) 모션벡터
            result[y][x] = np.transpose(v_T)
        except:
            result[y][x] = np.array([0, 0])

# 이미지에 화살표를 그려준다.
# background = cv2.addWeighted(color_first_img, 0.5, color_second_img, 0.5, 0)
background = color_first_img
for y in range(height):
    for x in range(width):
        if not np.array_equal(result[y][x], np.array([0, 0])):
            v, u = result[y][x]
            start_point = (x, y)
            end_point = (int(x + u), int(y + v))
            background = cv2.arrowedLine(img=background, pt1=start_point, pt2=end_point,
                                             color=(0, 0, 255), thickness=1, tipLength=0.1)


cv2.imshow('result', background)
cv2.waitKey()





