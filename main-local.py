from exif import Image
from datetime import datetime
import cv2
import os
import math
from math import sin, cos, sqrt, atan2, radians, atan, asin
import matplotlib.pyplot as plt
import time
import numpy as np

start_runtime = time.time()

# thresholds:

upper_black = 35 #35 - mai am de cautat (ar putea fi instabil)
threshold_brightness = 20
threshold_color = 50
threshold_speed = 0.125 #3

# image attributes:

earth_GSD = 25296
#12648 pt 4k
#39580 pt 1296×972
#26717 pt 1920x1440
#25296 pt 2028x1520 sau 50592

block_x, block_y = 910, 0
block_height = 1520
block_width = 380

img_width = 2028
img_height = 1520
center = (img_width / 2, img_height / 2)

to_be_ignored = np.zeros((img_height, img_width), dtype = int)
for l in range(block_y, block_height):
    to_be_ignored[l][block_x:block_x + block_width] = np.ones(block_width, dtype = int)

# measurements:
ecuator_radius = 6378
pole_radius = 6357

# for cloud classification:
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

def valid(l, c):
    if l >= 0 and c >= 0 and l < img_height and c < img_width:
        return True
    return False

def bfs(img, isCloud, l, c, threshold, flag, empty):
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
            if valid(next_l, next_c) and isCloud[next_l][next_c] == empty and img[next_l][next_c] >= threshold:
                q.append((next_l, next_c))
                isCloud[next_l][next_c] = flag
                area += 1
    return area

def precalc_sequence(img, threshold):
    precalc = np.zeros((img_height, img_width), dtype = int)
    for l in range(0, img_height, interpolation_cloud_detection_mini):
        for c in range(0, img_width, interpolation_cloud_detection_mini):
            for i in range(0, interpolation_cloud_detection_mini):
                for j in range(0, interpolation_cloud_detection_mini):
                    if img[l + i][c + j] >= threshold:
                        precalc[l][c] += 1
    return precalc

def get_clouds(img):
    img = np.array(img)
    
    no_clouds = 0
    clouds = np.zeros((img_height, img_width), dtype = int)
    for l in range(0, img_height):
        for c in range(0, img_width):
            if img[l][c] >= lower_white_cloud_detection and clouds[l][c] == 0:
                no_clouds += 1
                area = bfs(img, clouds, l, c, lower_white_cloud_detection, no_clouds, 0)
                if area < area_cloud_detection * 2:
                    bfs(img, clouds, l, c, lower_white_cloud_detection, -1, no_clouds)
    
    precalc = precalc_sequence(clouds, 1)
    new_q = []
    for l in range(0, img_height - block_size_cloud_detection_second, interpolation_cloud_detection):
        for c in range(0, img_width - block_size_cloud_detection_second, interpolation_cloud_detection):
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
                    if valid(l - 1, c + j) == True:
                        new_q.append((l - 1, c + j))
                    if valid(l + block_size_cloud_detection_second, c + j) == True:
                        new_q.append((l + block_size_cloud_detection_second, c + j))
                for i in range(0, block_size_cloud_detection_second):
                    if c - 1 >= 0:
                        new_q.append((l + i, c - 1))

                    clouds[l + i][c:c + block_size_cloud_detection_second] = np.zeros(block_size_cloud_detection_second, dtype = int)
                    
                    if c + block_size_cloud_detection_second < img_width:
                        new_q.append((l + i, c + block_size_cloud_detection_second))

    cloud_type = np.zeros((img_height, img_width), dtype = int)
    no_clouds = 0
    for (l, c) in new_q:
        if cloud_type[l][c] == 0 and clouds[l][c] >= 1:
            no_clouds += 1
            area = bfs(clouds, cloud_type, l, c, 1, no_clouds, 0)
            if area < area_cloud_detection * 2:
                bfs(clouds, cloud_type, l, c, 1, -1, no_clouds)
    
    for l in range(0, img_height):
        for c in range(0, img_width):
            if cloud_type[l][c] > 1:
                cloud_type[l][c] = 1
            if img[l][c] >= lower_white_cloud_detection and cloud_type[l][c] <= 0:
                cloud_type[l][c] = 2
            elif img[l][c] < lower_white_cloud_detection:
                cloud_type[l][c] = 3
    
    return cloud_type

def get_time(image):
    with open(image, 'rb') as image_file:
        time_str = Image(image_file).get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(img1, img2):
    return (get_time(img2) - get_time(img1)).seconds

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    image_1_cv = cv2.resize(image_1_cv, (img_width, img_height))
    image_2_cv = cv2.resize(image_2_cv, (img_width, img_height))
    return image_1_cv, image_2_cv

def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number, scaleFactor = 2, WTA_K = 4)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

def calculate_matches_flann(descriptors_1, descriptors_2):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks = 300) #200
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k = 2)
    matches = list(filter(lambda x: len(x) == 2, matches))
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)
    return good

def calculate_matches(descriptors_1, descriptors_2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_1, descriptors_2, k = 2)
    matches = list(filter(lambda x: len(x) == 2, matches))
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)
    return good

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', match_img)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

def color_diff(rgb1, rgb2):
    weights = [2, 2, 1]
    sum = 0
    for i in range(0, 3):
        sum += abs(int(rgb1[i]) - int(rgb2[i])) * weights[i]
    return sum / 3

def not_black(p1, p2):
    return p1 >= upper_black and p2 >= upper_black

def find_matching_coordinates(img1, keypoints_1, img2, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    img1cv, img2cv = convert_to_cv(img1, img2)
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (img_width, img_height))
    img2 = cv2.resize(img2, (img_width, img_height))
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2) 
        if not_black(img1cv[y1, x1], img2cv[y2, x2]) and abs(int(img1cv[y1, x1]) - int(img2cv[y2, x2])) <= threshold_brightness and color_diff(img1[y1, x1], img2[y2, x2]) <= threshold_color:
            coordinates_1.append((x1,y1))
            coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def not_to_ignore(coords):
    if to_be_ignored[coords[0][1]][coords[0][0]] == 0 and to_be_ignored[coords[1][1]][coords[1][0]] == 0:
        return True
    return False

def get_altitude_per_cloud_type(ctype, lat):
    if ctype == 1:
        if -23.5 <= lat and lat <= 23.5:
            return 8
        if (23.5 <= lat and lat <= 66.5) or (-66.5 <= lat and lat <= -23.5):
            return 5
        if (66.5 <= lat and lat <= 90) or (-90 <= lat and lat <= -66.5):
            return 3
    if ctype == 2:
        if -23.5 <= lat and lat <= 23.5:
            return 6
        if (23.5 <= lat and lat <= 66.5) or (-66.5 <= lat and lat <= -23.5):
            return 4
        if (66.5 <= lat and lat <= 90) or (-90 <= lat and lat <= -66.5):
            return 2
    if ctype == 3:
        return 0.000
    return 0

def calculate_mean_distance(coordinates_1, coordinates_2, time_diff, lat1, lat2):
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    pairs = []
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        if distance / time_diff >= threshold_speed and not_to_ignore(coordinate) == True:
            alt1 = get_altitude_per_cloud_type(clouds_1[coordinate[0][1]][coordinate[0][0]], lat1)
            alt2 = get_altitude_per_cloud_type(clouds_2[coordinate[1][1]][coordinate[1][0]], lat2)
            pairs.append((distance, (alt1 + alt2) / 2, coordinate[0], coordinate[1]))
    return pairs

def get_earth_radius(latitude):
    return sqrt((((ecuator_radius ** 2) * cos(latitude)) ** 2 + ((pole_radius ** 2) * sin(latitude)) ** 2) / ((ecuator_radius * cos(latitude)) ** 2 + (pole_radius * sin(latitude)) ** 2))

def get_GSD(alt, earth_GSD):
    return (421 - alt) * earth_GSD / 421

def calculate_speed_in_kmps(feature_distance, earth_GSD, time_difference, earth_radius, alt, min_dist):
    GSD = get_GSD(alt, earth_GSD)
    distance = feature_distance * GSD / 100000
    distance = distance / sqrt(1 - (distance ** 2) / ((421 - alt) ** 2 + min_dist ** 2))
    #distance = 2 * asin(distance / (2 * earth_radius)) * (earth_radius + 408) #423
    distance = 2 * asin(distance / (2 * (earth_radius + alt))) * (earth_radius + 421) #423
    #distance = distance * (earth_radius + 421) / (earth_radius + alt )
    #distance = 2 * asin(distance / (2 * earth_radius)) * iss_height
    #distance = 2 * asin(distance / (2 * 6373)) * (6373 + 408)
    #distance = distance / earth_radius * (earth_radius + iss_height)
    speed = distance / time_difference
    return speed

def get_coords_raw(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lat = img.get("gps_latitude")
        lat_dir = img.get("gps_latitude_ref")
        long = img.get("gps_longitude")
        long_dir = img.get("gps_longitude_ref")
    return lat, lat_dir, long, long_dir

def get_coords(image):
    coords = get_coords_raw(image)
    lat = deg_min_sec_to_dec_converter(coords[0][0], coords[0][1], coords[0][2], coords[1])
    long = deg_min_sec_to_dec_converter(coords[2][0], coords[2][1], coords[2][2], coords[3])
    return lat, long

def deg_min_sec_to_dec_converter(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'E' or direction == 'S':
        dd *= -1
    return dd

def distance_between_coords(lat1, long1, lat2, long2):
    dlong = long2 - long1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius * c

def calculate_speed_from_coords(lat1, long1, lat2, long2, time_difference):
    return distance_between_coords(lat1, long1, lat2, long2) / time_difference

avg_speed = 0
avg_time = 0

folder_str = './pics-database/set3/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
images = [folder_str + img for img in images]
images.sort(key=get_time)

#cloud_detect_img = images
#cloud_detect_img = [img[:21] + 'cloud_detection/' + img[21:] for img in images]

#image_1 = images[0]
#image_2 = images[1]
#time_difference = get_time_difference(image_1, image_2) # Get time difference between images
#image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
#image_1_cv = cv2.resize(image_1_cv, (img_width, img_height))
#image_2_cv = cv2.resize(image_2_cv, (img_width, img_height))

#clouds_1 = get_clouds(image_1_cv)
#clouds_2 = get_clouds(image_2_cv)

#f, axarr = plt.subplots(2,1)
#axarr[0].imshow(clouds_1, 'gray')
#axarr[1].imshow(clouds_2, 'gray')
#plt.show()

count = 0
for i in range(1, len(images)):
    image_1 = images[i - 1]
    image_2 = images[i]
    time_difference = get_time_difference(image_1, image_2) # Get time difference between images
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
    
    clouds_1 = get_clouds(image_1_cv)
    clouds_2 = get_clouds(image_2_cv)
    
    #cv2.imwrite(cloud_detect_img[i - 1], clouds_1)
    #cv2.imwrite(cloud_detect_img[i], clouds_2)
    
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 10000) # Get keypoints and descriptors
    matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
    
    coordinates_1, coordinates_2 = find_matching_coordinates(image_1, keypoints_1, image_2, keypoints_2, matches)
    #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches
    lat1, long1 = get_coords(image_1)
    lat2, long2 = get_coords(image_2)
    lat = (lat1 + lat2) / 2
    
    pairs = calculate_mean_distance(coordinates_1, coordinates_2, time_difference, lat1, lat2)
    
    earth_radius_1 = get_earth_radius(lat1)
    earth_radius_2 = get_earth_radius(lat2)
    earth_radius = (earth_radius_1 + earth_radius_2) / 2
    
    speed_img = 0
    for (distance, alt, p1, p2) in pairs:
        min_dist = min(math.hypot(p1[0] - center[0], p1[1] - center[1]), math.hypot(p2[0] - center[0], p2[1] - center[1]))
        speed_for_match = calculate_speed_in_kmps(distance, earth_GSD, time_difference, earth_radius, alt, min_dist)
        speed_img += speed_for_match
    speed_img /= len(pairs)
    print(i - 1, i, speed_img, len(pairs))
    if speed_img != -1:
        tan_alfa = (lat2 - lat1) / (long2 - long1)
        cos_alfa = cos(atan(tan_alfa))
 
        earth_speed = cos(math.pi * ((lat1 + lat2) / 2) / 90) * earth_radius * (math.pi * 2 / (24 * 60 * 60))
        
        #earth_speed = (math.pi * 2 / (24 * 60 * 60)) * sqrt(ecuator_radius ** 2 - ((lat / 90) * pole_radius) ** 2)
        
        speed = sqrt(speed_img ** 2 + earth_speed ** 2 + 2 * cos_alfa * speed_img * earth_speed)
        
        vertical_speed = (421 + earth_radius_2 - earth_radius_1 - 421) / time_difference
        
        speed = sqrt(speed ** 2 + vertical_speed ** 2)
        
        avg_speed += speed
        count += 1
    avg_time += time_difference

avg_speed = avg_speed / count
avg_time = avg_time / (len(images) - 1)

print(avg_time)
print(time.time() - start_runtime)
print(avg_speed)