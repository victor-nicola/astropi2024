import os

width = '100'
height = '380'
x = '225'
y = '0'

folder_str = 'positives/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
images = [folder_str + img for img in images]

file = open('positives.dat', 'w+')
for i in range(0, len(images)):
    file.write(images[i] + '  1  ' + x + ' ' + y + ' ' + width + ' ' + height + '\n')
    #file.write(images[i] + '\n')

file.close()