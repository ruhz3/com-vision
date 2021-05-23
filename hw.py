import cv2
import numpy as np


SOBEL_Y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]
SOBEL_X = [[1, 0, -1],
           [2, 0, -2],
           [1, 0, -1]]

N_SIZE = 3

# <editor-fold desc="컨벌루션 함수">
def conv(image, kernel, padding=1, strides=1):
    xImgShape, yImgShape = image.shape
    xKernShape, yKernShape = kernel.shape

    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)  # 59
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)  # 32
    output = np.zeros((xOutput, yOutput), np.uint8)

    if padding != 0:
        imagePadded = np.zeros((xImgShape + padding*2, yImgShape + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image
    for y in range(yImgShape):
        if y % strides == 0:
            for x in range(xImgShape):
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output
# </editor-fold>


WIDTH = 600
HEIGHT = 400

# 이미지를 불러온다.
color_first_img = cv2.imread("img/1.png", 1)
first_img = cv2.imread("img/1.png", 0)
second_img = cv2.imread("img/2.png", 0)
color_first_img = cv2.resize(color_first_img, (WIDTH, HEIGHT))
first_img = cv2.resize(first_img, (WIDTH, HEIGHT))
second_img = cv2.resize(second_img, (WIDTH, HEIGHT))
height, width = first_img.shape

# df/dy, df/dx, df/dt를 찾는다.
df_dy = conv(first_img, np.array(SOBEL_Y))
df_dx = conv(first_img, np.array(SOBEL_X))
df_dt = first_img - second_img

# A, b를 구성하고, v를 구한다.
gap = int(N_SIZE/2)
result = np.empty(shape=(height, width, 2))
for y in range(1, height-1):
    for x in range(1, width-1):
        try:
            y_tmp = df_dy[y - gap:y + gap + 1, x - gap:x + gap + 1]
            x_tmp = df_dx[y - gap:y + gap + 1, x - gap:x + gap + 1]
            t_tmp = df_dt[y - gap:y + gap + 1, x - gap:x + gap + 1]

            A = cv2.hconcat([np.reshape(y_tmp, (N_SIZE ** 2, 1)), np.reshape(x_tmp, (N_SIZE ** 2, 1))])
            A_T = np.transpose(A)
            b = -np.reshape(t_tmp, (N_SIZE ** 2, 1))
            v_T = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_T, A)), A_T), b)

            result[y][x] = np.transpose(v_T)
        except:
            result[y][x] = np.array([0, 0])

for y in range(height):
    for x in range(width):
        if not np.array_equal(result[y][x], np.array([0, 0])):
            color_first_img[y][x][0] = 0
            color_first_img[y][x][1] = 0
            color_first_img[y][x][2] = 255

cv2.imshow('dy', df_dy)
cv2.imshow('dx', df_dx)
cv2.imshow('result', color_first_img)
cv2.waitKey()




