import cv2
from matplotlib import pyplot as plt

# Opening image
img = cv2.imread("./pics-database/set10/theq_0278_53247635441_o.jpg")
img = cv2.resize(img, (507, 380))
   
# OpenCV opens images as RGB 
# but we want it as RGB We'll 
# also need a grayscale version
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Use minSize because for not 
# bothering with extra-small 
# dots that would look like STOP signs
stop_data = cv2.CascadeClassifier('./train-opencv-model/model/cascade.xml')
   
found = stop_data.detectMultiScale(img_gray)
   
# Don't do anything if there's 
# no sign
amount_found = len(found)
print(amount_found)

if amount_found != 0:
       
    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:
        print(x, y, width, height)
        #y = 0
        #width = 70
        #height = 380
        
        # We draw a green rectangle around
        # every recognized sign
        cv2.rectangle(img_rgb, (x*1, y*1), ((x + width)*1, (y + height)*1), (0, 255, 0), 1)
else:
    print('none')
# Creates the environment of 
# the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()