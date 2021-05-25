import cv2
import numpy as np


""" param 조작 ↓"""
N_SIZE = 7
GAUSSIAN_KSIZE = 7
DERIVATIVE_X = [[-1/2, 0, 1/2]]
DERIVATIVE_Y = [[-1/2],
                [0],
                [1/2]]

# 이미지를 불러온다.
color_first_img = cv2.imread("img/1.png", 1)
color_second_img = cv2.imread("img/2.png", 1)
first_img = cv2.imread("img/1.png", 0)
second_img = cv2.imread("img/2.png", 0)

# 가우시안 필터를 적용한다.
if GAUSSIAN_KSIZE != 0:
    first_img = cv2.GaussianBlur(first_img, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)
    second_img = cv2.GaussianBlur(second_img, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)

# df/dy, df/dx, df/dt를 찾는다.
height, width = first_img.shape
df_dy = cv2.filter2D(first_img, ddepth=cv2.CV_64F, kernel=np.array(DERIVATIVE_Y))
df_dx = cv2.filter2D(first_img, ddepth=cv2.CV_64F, kernel=np.array(DERIVATIVE_X))
df_dt = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        df_dt[y][x] = float(second_img[y][x]) - float(first_img[y][x])

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

            # 계산해 v_T를 구한다.
            mat1 = np.matmul(A_T, A)
            mat2 = np.matmul(A_T, b)
            v_T = np.matmul(np.linalg.inv(mat1), mat2)
            # result[y][x]에 저장한다.
            result[y][x] = np.transpose(v_T)
        except:
            # 역행렬이 없는 경우, 예외처리한다.
            result[y][x] = np.array([0., 0.])

# 이미지에 화살표를 그려준다.
background = cv2.addWeighted(color_first_img, 0.8, color_second_img, 0.2, 0)
for y in range(n, height-n, n):
    for x in range(n, width-n, n):
        v, u = result[y][x]
        start_point = (x, y)
        end_point = (x+int(u), y+int(v))
        if start_point != end_point:
            background = cv2.arrowedLine(img=background, pt1=start_point, pt2=end_point,
                                            color=(0, 0, 255), thickness=1, tipLength=0.1)

# 이미지를 보여준다.
cv2.imshow('result', background)
cv2.waitKey()





