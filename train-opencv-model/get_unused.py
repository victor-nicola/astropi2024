import os

file = open('positives.dat', 'r')
lines = file.readlines()
pics = []
for l in lines:
    pics.append(l[10:37])
# pics.sort()

# out = open('positives_sorted.dat', 'w')
# for i in range(0, len(pics)):
#     # print(pics[i])
#     out.write(pics[i] + '\n')

folder_str = '../pics-database/set10/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
for_testing = []
for img in images:
    if not img in pics:
        for_testing.append(img)
for_testing.sort()
for i in range(0, len(for_testing)):
    print(for_testing[i])
