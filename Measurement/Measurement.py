import cv2
import numpy as np
from PIL import Image



dir = "./cropped-removebg-preview (1).png"
img = cv2.imread(dir)
# img = cv2.flip(img, 0)
# img = cv2.resize(img, (500, 1000))
width, height = 400, 800
pil_img = Image.open(dir)
pil_img_size = pil_img.size
print(pil_img_size)
pts1 = np.float32([[210,5],[429,92],[7,449],[329,558]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width, height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img, matrix,(width,height))

for x in range(0,4):
    img = cv2.circle(img, (int(pts1[x][0]),int(pts1[x][1])), 5 , (0,0,255), cv2.FILLED)

cv2.imshow("warped",imgOutput)
cv2.waitKey(0)