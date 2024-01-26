from exif import Image
from datetime import datetime
import cv2
import os
import math
from math import sin, cos, sqrt, atan2, radians, atan

earth_radius = 6373.0
earthRotationCoef = 0.4630773148
GSD = 12648
#12648 pt 4k
#39580 pt 1296×972
expected_val = 7.66
threshold_low = -1.66
threshold_up = 100

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
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

def calculate_mean_kmps(coordinates_1, coordinates_2, GSD, time_difference):
    all_distances = avg_speed = no_values = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        speed = calculate_speed_in_kmps(distance, GSD, time_difference)
        if speed - expected_val > threshold_low and speed - expected_val < threshold_up:
            avg_speed += speed
            no_values += 1
    if no_values == 0:
        return 0
    return avg_speed / no_values

def get_coords(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lat = img.get("gps_latitude")
        lat_dir = img.get("gps_latitude_ref")
        long = img.get("gps_longitude")
        long_dir = img.get("gps_longitude_ref")
    return lat, lat_dir, long, long_dir

def deg_min_sec_to_dec_converter(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    #if direction == 'E' or direction == 'S':
        #dd *= -1
    return dd;

def distance_between_coords(lat1, long1, lat2, long2):
    dlong = long2 - long1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius * c

def calculate_speed_from_coords(lat1, long1, lat2, long2, time_difference):
    return distance_between_coords(lat1, long1, lat2, long2) / time_difference

avg_speed_img = avg_speed = avg_time = no_values = 0

folder_str = './pics-database/set9/'
images = list(filter(lambda file: file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'), os.listdir(folder_str)))
#img_str = image_1 = image_2 = './pics-database/set9/'
#image_1 = img_str + str(1) + '.jpg'
image_1 = folder_str + images[0]

#for id in range(2, 47): # range(1743, 1762)
for id in range(1, len(images)):
    #image_2 = img_str + str(id) + '.jpg'
    image_2 = folder_str + images[id]
    time_difference = get_time_difference(image_1, image_2) # Get time difference between images
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
    
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 8000) # Get keypoints and descriptors
    matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
    #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches
    
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    #average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    speed_img = calculate_mean_kmps(coordinates_1, coordinates_2, GSD, time_difference)
    
    coords_1 = get_coords(image_1)
    coords_2 = get_coords(image_2)
    
    lat1 = deg_min_sec_to_dec_converter(coords_1[0][0], coords_1[0][1], coords_1[0][2], coords_1[1])
    long1 = deg_min_sec_to_dec_converter(coords_1[2][0], coords_1[2][1], coords_1[2][2], coords_1[3])
    
    lat2 = deg_min_sec_to_dec_converter(coords_2[0][0], coords_2[0][1], coords_2[0][2], coords_2[1])
    long2 = deg_min_sec_to_dec_converter(coords_2[2][0], coords_2[2][1], coords_2[2][2], coords_2[3])
    
    tan_alfa = (lat2 - lat1) / (long2 - long1)
    cos_alfa = cos(atan(tan_alfa))

    earth_speed = cos(math.pi * ((lat1 + lat2) / 2) / 90) * earth_radius * (math.pi * 2 / (24 * 60 * 60))

    speed = sqrt(speed_img ** 2 + earth_speed ** 2 + 2 * cos_alfa * speed_img * earth_speed)
    
    avg_time += time_difference
    if speed - expected_val > threshold_low and speed - expected_val < threshold_up:
        avg_speed_img += speed_img
        avg_speed += speed
        no_values += 1
    
    image_1 = image_2

avg_speed_img = avg_speed_img / no_values
avg_speed = avg_speed / no_values
avg_time = avg_time / (len(images) - 1)
#print((avg_speed_img + avg_speed_coords) / 2 + earthRotationCoef)

print(avg_speed_img)
print(avg_speed)

#x = 455.8176
#print(sqrt(math.pi * 2 * (earth_radius**2) * (1 - (sqrt(earth_radius**2 - x**2) / earth_radius))))
