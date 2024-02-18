import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# NU TE ATINGE DE NICIO MARIME SI NICIUN INTERPOLATION CA ALTFEL LITERALMENTE NU MAI MERGE NIMIC

height_cloud_detection = 1520 #3040
width_cloud_detection = 2028 #4056
block_size_cloud_detection = 64 #128
area_cloud_detection = block_size_cloud_detection ** 2

block_size_cloud_detection_second = 64 #128
interpolation_cloud_detection = 16 #32
mini_block_size_cloud_detection = 8 #16
interpolation_cloud_detection_mini = 4 #8
smallest_area = (interpolation_cloud_detection_mini ** 2) * 4

lower_white_cloud_detection = 190

dir_l = [-1, 1,  0, 0]
dir_c = [ 0, 0, -1, 1]

def valid(l, c, nl, nc):
    if l >= 0 and c >= 0 and l < nl and c < nc:
        return True
    return False

def bfs(img, isCloud, l, c, nl, nc, threshold, flag, empty):
    q = [(l, c)]
    area = 1
    isCloud[l][c] = flag
    while len(q) > 0:
        qfront = q.pop(0)
        l = qfront[0]
        c = qfront[1]
        for d in range(0, len(dir_l)):
            next_l = l + dir_l[d]
            next_c = c + dir_c[d]
            if valid(next_l, next_c, nl, nc) and isCloud[next_l][next_c] == empty and img[next_l][next_c] >= threshold:
                q.append((next_l, next_c))
                isCloud[next_l][next_c] = flag
                area += 1
    return area

def precalc_sequence(img, threshold):
    precalc = np.zeros((height_cloud_detection, width_cloud_detection), dtype = int)
    for l in range(0, height_cloud_detection, interpolation_cloud_detection_mini):
        for c in range(0, width_cloud_detection, interpolation_cloud_detection_mini):
            for i in range(0, interpolation_cloud_detection_mini):
                for j in range(0, interpolation_cloud_detection_mini):
                    if img[l + i][c + j] >= threshold:
                        precalc[l][c] += 1
    return precalc

def get_clouds(img):
    found = []
    img = cv2.resize(img, (width_cloud_detection, height_cloud_detection))
    img = np.array(img)
    
    #first_filter = time.time()
    no_clouds = 0
    removed = []
    clouds = np.zeros((height_cloud_detection, width_cloud_detection), dtype = int)
    for l in range(0, height_cloud_detection):
        for c in range(0, width_cloud_detection):
            if img[l][c] >= lower_white_cloud_detection and clouds[l][c] == 0:
                no_clouds += 1
                area = bfs(img, clouds, l, c, height_cloud_detection, width_cloud_detection, lower_white_cloud_detection, no_clouds, 0)
                if area < area_cloud_detection * 2:
                    bfs(img, clouds, l, c, height_cloud_detection, width_cloud_detection, lower_white_cloud_detection, -1, no_clouds)
    
    #print('first_filter: ' + str(time.time() - first_filter))

    #fix aici ar merge optimizare cu precalculare
    #optimize = time.time()
    precalc = precalc_sequence(clouds, 1)
    new_q = []
    for l in range(0, height_cloud_detection - block_size_cloud_detection_second, interpolation_cloud_detection):
        for c in range(0, width_cloud_detection - block_size_cloud_detection_second, interpolation_cloud_detection):
            counter = 0
            for i in range(0, block_size_cloud_detection_second, interpolation_cloud_detection_mini):
                for j in range(0, block_size_cloud_detection_second, interpolation_cloud_detection_mini):
                    white = precalc[l + i][c + j] + precalc[l + i][c + j + interpolation_cloud_detection_mini]
                    white += precalc[l + i + interpolation_cloud_detection_mini][c + j] + precalc[l + i + interpolation_cloud_detection_mini][c + j + interpolation_cloud_detection_mini]
                    ratio = white / smallest_area
                    if 0.30 <= ratio and ratio <= 0.7:
                        counter += 1
            if counter > ((block_size_cloud_detection_second / mini_block_size_cloud_detection) ** 2) * 0.6:
                for j in range(-1, block_size_cloud_detection_second + 1):
                    if valid(l - 1, c + j, height_cloud_detection, width_cloud_detection) == True:
                        new_q.append((l - 1, c + j))
                    if valid(l + block_size_cloud_detection_second, c + j, height_cloud_detection, width_cloud_detection) == True:
                        new_q.append((l + block_size_cloud_detection_second, c + j))
                for i in range(0, block_size_cloud_detection_second):
                    if c - 1 >= 0:
                        new_q.append((l + i, c - 1))

                    clouds[l + i][c:c + block_size_cloud_detection_second] = np.zeros(block_size_cloud_detection_second, dtype = int)
                    
                    if c + block_size_cloud_detection_second < width_cloud_detection:
                        new_q.append((l + i, c + block_size_cloud_detection_second))
    #print('optimize: ' + str(time.time() - optimize))
    # facem curat
    #clean = time.time()
    isCloud = np.zeros((height_cloud_detection, width_cloud_detection), dtype = int)
    no_clouds = 0
    for (l, c) in new_q:
        if isCloud[l][c] == 0 and clouds[l][c] >= 1:
            no_clouds += 1
            area = bfs(clouds, isCloud, l, c, height_cloud_detection, width_cloud_detection, 1, no_clouds, 0)
            if area < area_cloud_detection * 2:
                bfs(clouds, isCloud, l, c, height_cloud_detection, width_cloud_detection, 1, -1, no_clouds)
    
    #for l in range(0, height_cloud_detection):
     #   for c in range(0, width_cloud_detection):
      #      if isCloud[l][c] > 0:
       #         isCloud[l][c] = 1
        #    elif isCloud[l][c] < 0:
         #       isCloud[l][c] = 0
    
    return isCloud

img = cv2.imread("positives/frame01871_53238819088_o.jpg", 0)

start = time.time()
clouds = get_clouds(img)

print(time.time() - start)

plt.subplot(1, 1, 1)
plt.imshow(clouds, cmap = 'gray')
plt.show()