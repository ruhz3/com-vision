import cv2
import numpy as np
from math import sqrt

DEBUG = False

N_SIZE = 3/
THRESHOLD = 0.01
GAUSSIAN_KSIZE = 0

DERIVATIVE_X = [[0, 1, -1]]
DERIVATIVE_Y = [[0], [1], [-1]]

# 이미지를 불러온다.
color_first_img = cv2.imread("img/11.png", 1)
color_second_img = cv2.imread("img/22.png", 1)
first_img = cv2.imread("img/11.png", 0)
second_img = cv2.imread("img/22.png", 0)
if DEBUG:
    color_first_img = cv2.resize(color_first_img, (12, 8))
    color_second_img = cv2.resize(color_second_img, (12, 8))
    first_img = cv2.resize(first_img, (12, 8))
    second_img = cv2.resize(second_img, (12, 8))
height, width = first_img.shape

# df/dy, df/dx, df/dt를 찾는다.
if GAUSSIAN_KSIZE != 0:
    first_img = cv2.GaussianBlur(first_img, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)
    second_img = cv2.GaussianBlur(second_img, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)

if DEBUG:
    print("==================first_img==================")
    print(first_img)
    print("==================second_img==================")
    print(second_img)

df_dy = cv2.filter2D(first_img, ddepth=cv2.CV_64F, kernel=np.array(DERIVATIVE_Y))
df_dx = cv2.filter2D(first_img, ddepth=cv2.CV_64F, kernel=np.array(DERIVATIVE_X))
df_dt = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        df_dt[y][x] = int(second_img[y][x]) - int(first_img[y][x])

if DEBUG:
    print("==================df_dy==================")
    print(df_dy)
    print("==================df_dx==================")
    print(df_dx)
    print("==================df_dt==================")
    print(df_dt)

# A, b를 구성하고, v를 구한다.
n = int(N_SIZE / 2)
result = np.empty(shape=(height, width, 2))
eigen = np.empty(shape=(height, width))
for y in range(n, height - n):
    for x in range(n, width - n):
        try:
            # N 범위에 있는 픽셀들을 모아준다.
            y_tmp = df_dy[(y - n):(y + n + 1), (x - n):(x + n + 1)]
            x_tmp = df_dx[(y - n):(y + n + 1), (x - n):(x + n + 1)]
            t_tmp = df_dt[(y - n):(y + n + 1), (x - n):(x + n + 1)]
            # A, v, b 행렬을 구성한다.
            A = cv2.hconcat([np.reshape(y_tmp, (N_SIZE ** 2, 1)), np.reshape(x_tmp, (N_SIZE ** 2, 1))])
            A_T = np.transpose(A)
            b = -np.reshape(t_tmp, (N_SIZE ** 2, 1))

            mat1 = np.matmul(A_T, A)
            mat2 = np.matmul(A_T, b)

            eigen[y][x] = min(np.linalg.eigvals(mat1))
            v_T = np.matmul(np.linalg.inv(mat1), mat2)
            # 결과 (v, u) 모션벡터
            result[y][x] = np.transpose(v_T)
            if DEBUG:
                print(f"##################{[y, x]}###################")
                print(f'y_tmp : {y_tmp}')
                print(f'x_tmp : {x_tmp}')
                print(f't_tmp : {t_tmp}')
                print(f'A : {A}')
                print(f'A_T : {A_T}')
                print(f'b : {b}')
                print(f'mat1 : {mat1}')
                print(f'mat2 : {mat2}')
                print(f'min(eigen) : {eigen[y][x]}')
                print(f'v_T : {v_T}')
        except:
            result[y][x] = np.array([0, 0])

# 이미지에 화살표를 그려준다.
background = cv2.addWeighted(color_first_img, 0.5, color_second_img, 0.5, 0)
for y in range(height):
    for x in range(width):
        if not np.array_equal(result[y][x], np.array([0, 0])):
            v, u = result[y][x]
            start_point = (x, y)
            if eigen[y][x] < THRESHOLD:
                end_point = (int(x - u*10), int(y - v*10))
                background = cv2.arrowedLine(img=background, pt1=start_point, pt2=end_point,
                                             color=(0, 0, 255), thickness=1, tipLength=0.1)


cv2.imshow('result', background)
cv2.waitKey()





