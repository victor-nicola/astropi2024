import os
import cv2

width = '740'
height = '3040'
x = '1810'
y = '0'

folder_str = 'positives/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
images = [folder_str + img for img in images]

file = open('positives.dat', 'w+')
for img in images:
    file.write(img + '  1  ' + x + ' ' + y + ' ' + width + ' ' + height + '\n')

print(cv2)