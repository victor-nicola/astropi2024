import os
import cv2

width = 2028
height = 1520

img = './pics-database/set10/theq_0208_53248130805_o.jpg'
img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (width, height))
cv2.imwrite('coaie.jpg', img)