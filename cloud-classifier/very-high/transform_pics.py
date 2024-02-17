import os
import cv2

width = 507
height = 380
lower_white = 190

def convert(img):
    for l in range(0, height):
        for c in range(0, width):
            if img[l, c] <= lower_white:
                img[l, c] = 0
    return img

folder_str = 'positives/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
images = [folder_str + img for img in images]

for path in images:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height))
    img = convert(img)
    cv2.imwrite(path, img)