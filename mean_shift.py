import cv2
import numpy as np
import math
E = 1
H_SIZE = 5
PADDING = 1


# 이미지를 가져온다.
grey_img = cv2.imread("img/1.png", 0)
color_img = cv2.imread("img/1.png", 1)
height, width = grey_img.shape
b, g, r = cv2.split(color_img)

h = int(H_SIZE/2)

dist_kernel = np.zeros((H_SIZE, H_SIZE))
dist_sum = 0
for y in range(H_SIZE):
    for x in range(H_SIZE):
        d = math.sqrt((y-h)**2 +(x-h)**2)
        dist_kernel[y][x] = d
        dist_sum += d
dist_kernel = dist_kernel/dist_sum

color = np.zeros((height, width))
for y in range(height):
    for x in range(width):



