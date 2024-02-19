if __name__ == "__main__":
    from exif import Image
    from datetime import datetime
    import cv2
    import math
    from math import sin, cos, sqrt, atan2, atan, asin
    import time
    from time import sleep
    from picamera import PiCamera
    from orbit import ISS
    import numpy as np
    
    # starting the counter to stop the program on time
    start_runtime = time.time()
    
    # setting up global variables and constants:
    
    # thresholds:
    upper_black = 35 #35 - mai am de cautat (ar putea fi instabil)
    threshold_brightness = 20
    threshold_color = 50
    threshold_speed = 0.125 #3
    
    # image attributes:
    earth_GSD = 25296
    
    # 12648 pt 4k
    # 39580 pt 1296×972
    # 26717 pt 1920x1440
    # 25296 pt 2028x1520 sau 50592
    
    block_x, block_y = 910, 0
    block_height = 1520
    block_width = 380
    
    img_width = 2028
    img_height = 1520
    center = (img_width / 2, img_height / 2)
    
    # as seen in certain tests and sample pictures from the last year, sometimes there were some obstructions in the middle of the image
    # therefore, we simply ignore any matched points within this area
    to_be_ignored = np.zeros((img_height, img_width), dtype = int)
    for l in range(block_y, block_height):
        to_be_ignored[l][block_x:block_x + block_width] = np.ones(block_width, dtype = int)
    
    # measurements:
    ecuator_radius = 6378
    pole_radius = 6357
    
    # for cloud classification:
    block_size_cloud_detection = 64 # 64
    area_cloud_detection = block_size_cloud_detection ** 2
    
    block_size_cloud_detection_second = 64 #64
    interpolation_cloud_detection = 16 #16
    mini_block_size_cloud_detection = 8 #8
    interpolation_cloud_detection_mini = 4 #4
    smallest_area = (interpolation_cloud_detection_mini ** 2) * 4
    
    lower_white_cloud_detection = 175
    
    # direction arrays
    dir_l = [-1, 1,  0, 0]
    dir_c = [ 0, 0, -1, 1]
    
    # this function sees if the coordinates are inside the image
    def valid(l, c):
        if l >= 0 and c >= 0 and l < img_height and c < img_width:
            return True
        return False
    
    # this is a generalized breadth-first search algorithm to fill in the clouds in the image
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
    
    # precalculating the amount of white in all of the 2d subarrays of the image sized interpolation_cloud_detection_mini x interpolation_cloud_detection_mini
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
        """
        This function returns a numpy 2d array as big as the image with the following proprieties:
        1 - if the pixel is in the highest type of cloud
        2 - if the pixel is in the second highest type of cloud
        3/0 - if the pixel is either a lower type of cloud or a body of water, or piece of land
        
        Firstly we filter out the pixels which are darker than a certain brightness threshold, remaining with possible clouds
        Secondly, we mark out every cloud using the bfs algorithm, and if its area is under a certain threshold, we mark this cloud as being the second highest type of clouds
        Then, we divide the image in submatrices of a smaller size, and the we further divide these submatrices into smaller submatrices
        Then, we count the precentage of white in these 2d arrays and in this way we can see if this submatrix contains scattered pieces of clouds => we classify them as the second highest type of clouds
        Lastly, the last procedure of taking out squares from the matrix can result in isolating some mini islands of clouds that should be removed as well => we repeat the first step with bfs
        """
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
                        if 0.30 <= ratio and ratio <= 0.75:
                            counter += 1
                if counter >= ((block_size_cloud_detection_second / mini_block_size_cloud_detection) ** 2) * 0.4:
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
    
    # getting the time from the image path file from exif
    def get_time(image):
        with open(image, 'rb') as image_file:
            time_str = Image(image_file).get("datetime_original")
            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time
    
    # getting the time difference between 2 time objects
    def get_time_difference(t1, t2):
        return (t2 - t1).seconds
    
    # getting the features from both of the pictures that we will match and further calculate distances and speeds from these
    def calculate_features(image_1, image_2, feature_number):
        orb = cv2.ORB_create(nfeatures = feature_number, scaleFactor = 2, WTA_K = 4)
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    # matches points in these images using a flann based matcher
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
    
    # matches points in these images using a brute force matcher
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
    
    # debugging function that shows the matches between 2 pictures
    def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
        match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
        cv2.imshow('matches', match_img)
        cv2.waitKey(0)
        cv2.destroyWindow('matches')
    
    # does a weighted average of the differences on the rgb color channels between 2 pixels in order to determine if the pixels are, in fact, matched correctly
    def color_diff(rgb1, rgb2):
        weights = [2, 2, 1]
        sum = 0
        for i in range(0, 3):
            sum += abs(int(rgb1[i]) - int(rgb2[i])) * weights[i]
        return sum / 3
    
    # determines if 2 points are both not black
    def not_black(p1, p2):
        return p1 >= upper_black and p2 >= upper_black
    
    # for every match, we verify its accuracy and if it's satisfying, we add it to an array of coordinates of matches
    def find_matching_coordinates(img1, img1cv, keypoints_1, img2, img2cv, keypoints_2, matches):
        coordinates_1 = []
        coordinates_2 = []
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
    
    # function that tests if the point in the image should be ignored (details on line 38)
    def not_to_ignore(coords):
        if to_be_ignored[coords[0][1]][coords[0][0]] == 0 and to_be_ignored[coords[1][1]][coords[1][0]] == 0:
            return True
        return False
    
    # get the altitude of a cloud based on its type and the latitude
    def get_altitude_per_cloud_type(ctype, lat):
        if ctype == 1:
            if -23.5 <= lat and lat <= 23.5:
                return 16
            if (23.5 <= lat and lat <= 66.5) or (-66.5 <= lat and lat <= -23.5):
                return 12
            if (66.5 <= lat and lat <= 90) or (-90 <= lat and lat <= -66.5):
                return 7
        if ctype == 2:
            if -23.5 <= lat and lat <= 23.5:
                return 13
            if (23.5 <= lat and lat <= 66.5) or (-66.5 <= lat and lat <= -23.5):
                return 10
            if (66.5 <= lat and lat <= 90) or (-90 <= lat and lat <= -66.5):
                return 5
        if ctype == 3:
            return 0.5
        return 0
    
    # takes the matches and if the speed of this match is under a threshold then it's a static object, most likely an obstruction so we just ignore it
    # if it's moving and not in the area that should be ignored, we return this pair: the distance of this match and the altitude of the matched points
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
    
    # getting the accurate earth radius based on the latitude, considering the fact that the earth is not a perfect circle
    def get_earth_radius(latitude):
        return sqrt((((ecuator_radius ** 2) * cos(latitude)) ** 2 + ((pole_radius ** 2) * sin(latitude)) ** 2) / ((ecuator_radius * cos(latitude)) ** 2 + (pole_radius * sin(latitude)) ** 2))
    
    # dinamically calculating the cm/pixel coeficient based on the coeficient at sea level and the altitude of the considered point
    def get_GSD(alt_point, alt_iss, earth_GSD):
        return (alt_iss - alt_point) * earth_GSD / alt_iss
    
    # calculating the speed of a match
    def calculate_speed_in_kmps(feature_distance, earth_GSD, time_difference, earth_radius, alt, alt_iss, min_dist):
        GSD = get_GSD(alt, alt_iss, earth_GSD)
        #GSD = earth_GSD
        distance = feature_distance * GSD / 100000
        #l = distance / sqrt(1 - (distance ** 2) / ((alt_iss - alt) ** 2 + min_dist ** 2))
        #distance = sqrt(l * distance)
        #distance = 2 * asin(distance / (2 * earth_radius)) * (earth_radius + 408) #423
        distance = 2 * asin(distance / (2 * (earth_radius + alt))) * (earth_radius + alt_iss) #423
        #distance = distance * (earth_radius + 421) / (earth_radius + alt )
        #distance = 2 * asin(distance / (2 * earth_radius)) * iss_height
        #distance = 2 * asin(distance / (2 * 6373)) * (6373 + 408)
        #distance = distance / earth_radius * (earth_radius + iss_height)
        speed = distance / time_difference
        return speed
    
    # getting the coordinates of a image from its path
    def get_coords_raw(image):
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            lat = img.get("gps_latitude")
            lat_dir = img.get("gps_latitude_ref")
            long = img.get("gps_longitude")
            long_dir = img.get("gps_longitude_ref")
        return lat, lat_dir, long, long_dir
    
    # getting the decimal degree format of the coords from the image exif
    def get_coords(image):
        coords = get_coords_raw(image)
        lat = deg_min_sec_to_dec_converter(coords[0][0], coords[0][1], coords[0][2], coords[1])
        long = deg_min_sec_to_dec_converter(coords[2][0], coords[2][1], coords[2][2], coords[3])
        return lat, long
    
    # converting an angle from the degrees minutes second and direction format to the decimal degree format
    def deg_min_sec_to_dec_converter(degrees, minutes, seconds, direction):
        dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
        if direction == 'E' or direction == 'S':
            dd *= -1
        return dd
    
    # calculates the distance in km between 2 coordinates
    def distance_between_coords(lat1, long1, lat2, long2):
        dlong = long2 - long1
        dlat = lat2 - lat1
    
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
        return earth_radius * c
    
    # calculating the speed in between 2 coords
    def calculate_speed_from_coords(lat1, long1, lat2, long2, time_difference):
        return distance_between_coords(lat1, long1, lat2, long2) / time_difference
    
    # getting the coordinates in the exif format
    def nice_coords(angle):
        sign, degrees, minutes, seconds = angle.signed_dms()
        exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
        return sign < 0, exif_angle
    
    # converting the angle in the decimal format
    def convert(angle):
        sign, degrees, minutes, seconds = angle.signed_dms()
        dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
        if sign < 0:
            dd *= -1
        return dd
    
    # taking a pic and registering its info: position, time, the image in grayscale and rgb
    def take_custom_pic(iss, camera, src):
        point = iss.coordinates()
    
        latitude = convert(point.latitude)
        longitude = convert(point.longitude)
        altitude = float(format(point.elevation.km))
        
        #print(altitude)
        
        south, exif_latitude = nice_coords(point.latitude)
        west, exif_longitude = nice_coords(point.longitude)
        
        camera.exif_tags['GPS.GPSLatitude'] = 1
        camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
        camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
        camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
        camera.exif_tags['GPS.GPSAltitude'] = (altitude, 1)
        
        camera.capture(src)
        t = get_time(src)
        image_rgb = cv2.imread(src)
        image_rgb = cv2.resize(image_rgb, (img_width, img_height))
        image_cv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        return latitude, longitude, altitude, t, image_cv, image_rgb
    
    def print_ans(avg_speed):
        # printing the result in the right format
        file = open('result.txt', 'w+')
        ans = str(avg_speed)
        file.write(ans[:6])
        file.close()
    
    # initialising the camera and giving it a delay to settle in
    camera = PiCamera()
    camera.resolution = (img_width, img_height)
    sleep(2)
    
    # setting up variables
    count = avg_speed = 0
    runtime = 10 * 60 - 40
    
    # taking the first pic, so that we can write to memory the least and move the proprieties from variable to variable (as seen on line )
    lat1, long1, alt1, t1, image_1_cv, image_1_rgb = take_custom_pic(ISS(), camera, '1.jpg')
    clouds_1 = get_clouds(image_1_cv)
    earth_radius_1 = get_earth_radius(lat1)
    
    # while the time hasn't passed yet
    while time.time() - start_runtime < runtime:
        # we take a picture
        lat2, long2, alt2, t2, image_2_cv, image_2_rgb = take_custom_pic(ISS(), camera, '2.jpg')
        clouds_2 = get_clouds(image_2_cv)
        
        time_difference = get_time_difference(t1, t2) + 0.001 # Get time difference between images, + constant accounting for the writing speed on the sd card
        
        keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 10000) # Get keypoints and descriptors
        matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
        
        coordinates_1, coordinates_2 = find_matching_coordinates(image_1_rgb, image_1_cv, keypoints_1, image_2_rgb, image_2_cv, keypoints_2, matches) # getting matching coords
        #display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches
        
        # calculating the earth radius in this position and calculating the speed for every match
        earth_radius_2 = get_earth_radius(lat2)
        earth_radius = (earth_radius_1 + earth_radius_2) / 2
        pairs = calculate_mean_distance(coordinates_1, coordinates_2, time_difference, lat1, lat2)
        alt_iss = (alt1 + alt2) / 2
        
        speed_img = 0
        if len(pairs) >= 10:
            for (distance, alt, p1, p2) in pairs:
                min_dist = min(math.hypot(p1[0] - center[0], p1[1] - center[1]), math.hypot(p2[0] - center[0], p2[1] - center[1]))
                speed_for_match = calculate_speed_in_kmps(distance, earth_GSD, time_difference, earth_radius, alt, alt_iss, min_dist)
                speed_img += speed_for_match
            speed_img /= len(pairs)
        
        print(speed_img, len(pairs))
        
        # if we have at least one match that we considered
        if speed_img != 0:
            # accounting for the earth rotation
            tan_alfa = (lat2 - lat1) / (long2 - long1)
            cos_alfa = cos(atan(tan_alfa))
    
            earth_speed = cos(math.pi * ((lat1 + lat2) / 2) / 90) * earth_radius * (math.pi * 2 / (24 * 60 * 60))
    
            speed = sqrt(speed_img ** 2 + earth_speed ** 2 + 2 * cos_alfa * speed_img * earth_speed)
            
            # accounting for the vertical speed component of the overall iss speed
            #vertical_speed = (alt2 + earth_radius_2 - earth_radius_1 - alt1) / time_difference
            
            #speed = sqrt(speed ** 2 + vertical_speed ** 2)
            
            # adding this value to our average
            avg_speed += speed
            count += 1
            print_ans(avg_speed / count)
        
        # transfering the current data to image_1 so that we don't have to duplicate calculations
        lat1, long1, alt1, earth_radius_1, t1, image_1_cv, image_1_rgb, clouds_1 = lat2, long2, alt2, earth_radius_2, t2, image_2_cv, image_2_rgb, clouds_2
    
    avg_speed = avg_speed / count
    
    print(time.time() - start_runtime)
    print(avg_speed)
    
    # printing the result in the right format
    print_ans(avg_speed)