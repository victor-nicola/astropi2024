import cv2
import numpy as np
from matplotlib import pyplot as plt

# types of clouds:
# 0 - very high
# 1 - high
# 2 - medium
# 3 - low

height_cloud_detection = 3040 #380
width_cloud_detection = 4056 #507
block_size_cloud_detection = 128 #20
area_cloud_detection = block_size_cloud_detection ** 2

block_size_cloud_detection_second = 128 #128
interpolation_cloud_detection = 32 #32
mini_block_size_cloud_detection = 16
interpolation_cloud_detection_mini = 8

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
    upper_left_corner = [l, c]
    lower_right_corner = [l, c]
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
                
                lower_right_corner[0] = max(lower_right_corner[0], next_l)
                lower_right_corner[1] = max(lower_right_corner[1], next_c)
                
                upper_left_corner[0] = min(upper_left_corner[0], next_l)
                upper_left_corner[1] = min(upper_left_corner[1], next_c)
                
                area += 1
    return upper_left_corner, lower_right_corner, area

def get_clouds(img):
    found = []
    img = cv2.resize(img, (width_cloud_detection, height_cloud_detection))
    img_np = np.array(img)
    for l in range(0, height_cloud_detection, block_size_cloud_detection):
        for c in range(0, width_cloud_detection, block_size_cloud_detection):
            nl = min(height_cloud_detection, l + block_size_cloud_detection)
            nc = min(width_cloud_detection, c + block_size_cloud_detection)
            
            #create submatrix
            sub_matrix = img_np[l:nl, c:nc]
            nl -= l
            nc -= c
            isCloud = [[False for i in range(0, nc)] for i in range(0, nl)]
            
            #get amount of white in image and amount of clouds
            white = 0
            no_clouds = 0
            for i in range(0, nl):
                for j in range(0, nc):                
                    if sub_matrix[i, j] >= lower_white_cloud_detection:
                        white += 1
                        if isCloud[i][j] == False:
                            no_clouds += 1
                            bfs(sub_matrix, isCloud, i, j, nl, nc, lower_white_cloud_detection, True, False)
            
            if no_clouds >= 1:
                cloud_type = 2
                ratio = white / area_cloud_detection            
                if ratio >= 0.5:
                    cloud_type = 0
                    if no_clouds >= 3:
                        cloud_type = 1
                elif ratio < 0.5 and no_clouds >= 3:
                    cloud_type = 1
                else:
                    cloud_type = 3 # tre sa vedem cum facem pt medium
                
                if cloud_type == 0:
                    found.append((c, l, nc, nl))
    
    clouds = np.zeros((height_cloud_detection, width_cloud_detection), dtype = int)
    boxes = []
    removed = []
    flag = 0
    for (c, l, nc, nl) in found:
        for i in range(0, nl):
            for j in range(0, nc):                
                if img[l + i, c + j] >= lower_white_cloud_detection and clouds[l + i][c + j] == 0:
                    flag += 1
                    upper_left_corner, lower_right_corner, area = bfs(img_np, clouds, l + i, c + j, height_cloud_detection, width_cloud_detection, lower_white_cloud_detection, flag, 0)
                    if area < area_cloud_detection * 2:
                        removed.append((upper_left_corner[1], upper_left_corner[0], lower_right_corner[1] - upper_left_corner[1], lower_right_corner[0] - upper_left_corner[0], flag))
                    else:
                        boxes.append((upper_left_corner[1], upper_left_corner[0], lower_right_corner[1] - upper_left_corner[1], lower_right_corner[0] - upper_left_corner[0]))
    
    for (c, l, nc, nl, flag) in removed:
        for i in range(0, nl):
            for j in range(0, nc):
                if clouds[l + i][c + j] == flag:
                    clouds[l + i][c + j] = 0
    
    ans = clouds
    for l in range(0, height_cloud_detection - block_size_cloud_detection_second, interpolation_cloud_detection):
        for c in range(0, width_cloud_detection - block_size_cloud_detection_second, interpolation_cloud_detection):
            counter = 0
            for i in range(0, block_size_cloud_detection_second, interpolation_cloud_detection_mini):
                for j in range(0, block_size_cloud_detection_second, interpolation_cloud_detection_mini):
                    white = 0
                    area = 0
                    for k in range(0, mini_block_size_cloud_detection):
                        for x in range(0, mini_block_size_cloud_detection):
                            if i + k < height_cloud_detection and j + x < width_cloud_detection:
                                area += 1
                                if ans[l + i + k][c + j + x] != 0: #clouds
                                    white += 1
                    ratio = white / area
                    if 0.30 <= ratio and ratio <= 0.7:
                        counter += 1
            if counter > ((block_size_cloud_detection_second / mini_block_size_cloud_detection) ** 2) * 0.6:
                for i in range(0, block_size_cloud_detection_second):
                    for j in range(0, block_size_cloud_detection_second):
                        ans[l + i][c + j] = 0
    
    for l in range(0, height_cloud_detection):
        for c in range(0, width_cloud_detection):
            if ans[l][c] != 0:
                ans[l][c] = 1
    
    # facem curat
    removed = []
    isCloud = np.zeros((height_cloud_detection, width_cloud_detection), dtype = int)
    no_clouds = 0
    for l in range(0, height_cloud_detection):
        for c in range(0, width_cloud_detection):
            if isCloud[l][c] == 0 and ans[l][c] == 1:
                no_clouds += 1
                c1, c2, area = bfs(ans, isCloud, l, c, height_cloud_detection, width_cloud_detection, 1, no_clouds, 0)
                if area < area_cloud_detection * 2:
                    removed.append(no_clouds)
    
    new_colors = [1 for i in range(0, no_clouds)]
    for r in removed:
        new_colors[r - 1] = 0
    
    for l in range(0, height_cloud_detection):
        for c in range(0, width_cloud_detection):
            ans[l][c] = new_colors[isCloud[l][c] - 1]
    
    return ans

img = cv2.imread("positives/frame01840_53238890544_o.jpg", 0)

clouds = get_clouds(img)

plt.subplot(1, 1, 1)
plt.imshow(clouds, cmap = 'gray')
plt.show()
