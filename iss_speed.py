from exif import Image
from datetime import datetime
import cv2
import os
import math
from math import sin, cos, sqrt, atan2, radians, atan, asin
import matplotlib.pyplot as plt
import time

start_runtime = time.time()

iss_height = 6793
upper_black = 35 #35 - mai am de cautat (ar putea fi instabil)
#earth_radius = 6373.0
ecuator_radius = 6378
pole_radius = 6357
earthRotationCoef = 0.4630773148
GSD = 12648
threshold_brightness = 20
threshold_color = 50
threshold_speed = 0.125 #3
block_x, block_y = 1700, 0
block_height = 3040
block_width = 900
#12648 pt 4k
#39580 pt 1296×972
#26717 pt 1920x1440

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

def calculate_mean_distance(coordinates_1, coordinates_2, time_diff):
    all_distances = still = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        if distance / time_diff < threshold_speed:
            still += 1
        else:
            all_distances = all_distances + distance
    if len(merged_coordinates) - still <= 0:
        return -1
    return all_distances / (len(merged_coordinates) - still)

def get_earth_radius(latitude):
    return sqrt((((ecuator_radius ** 2) * cos(latitude)) ** 2 + ((pole_radius ** 2) * sin(latitude)) ** 2) / ((ecuator_radius * cos(latitude)) ** 2 + (pole_radius * sin(latitude)) ** 2))

def calculate_speed_in_kmps(feature_distance, GSD, time_difference, earth_radius):
    distance = feature_distance * GSD / 100000
    distance = 2 * asin(distance / (2 * earth_radius)) * (earth_radius + 408) #423
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

avg_speed = avg_time = 0

folder_str = './pics-database/set9/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
images = [folder_str + img for img in images]
images.sort(key=get_time)

count = 0
for id in range(1, len(images)):
    image_1 = images[id - 1]
    image_2 = images[id]
    time_difference = get_time_difference(image_1, image_2) + 0.56 # Get time difference between images
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
    
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 10000) # Get keypoints and descriptors
    matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
    
    coordinates_1, coordinates_2 = find_matching_coordinates(image_1, keypoints_1, image_2, keypoints_2, matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2, time_difference)
    #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches
    lat1, long1 = get_coords(image_1)
    lat2, long2 = get_coords(image_2)
    
    earth_radius = (get_earth_radius(lat1) + get_earth_radius(lat2)) / 2
    speed_img = calculate_speed_in_kmps(average_feature_distance, GSD, time_difference, earth_radius)
    if speed_img != -1:
        tan_alfa = (lat2 - lat1) / (long2 - long1)
        cos_alfa = cos(atan(tan_alfa))

        earth_speed = cos(math.pi * ((lat1 + lat2) / 2) / 90) * earth_radius * (math.pi * 2 / (24 * 60 * 60))

        speed = sqrt(speed_img ** 2 + earth_speed ** 2 + 2 * cos_alfa * speed_img * earth_speed)
        
        avg_speed += speed
        count += 1
    avg_time += time_difference

avg_speed = avg_speed / count
avg_time = avg_time / (len(images) - 1)

print(avg_time)
print(time.time() - start_runtime)
print(avg_speed)