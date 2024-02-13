import os
import cv2

width = 507
height = 380

folder_str = './negatives/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
images = [folder_str + img for img in images]

for path in images:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height))
    cv2.imwrite(path, img)