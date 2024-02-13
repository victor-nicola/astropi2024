import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./pics-database/set10/theq_0278_53247635441_o.jpg")
img_resized = cv2.resize(img, (507, 380))

img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('./train-opencv-model/model/cascade.xml')
   
found = stop_data.detectMultiScale(img_gray, minSize = (20, 20))

amount_found = len(found)
print(amount_found)

if amount_found != 0:
    for (x, y, width, height) in found:
        print(x, y, width, height)
        cv2.rectangle(img_rgb, (x*8, y*8), ((x + width)*8, (y + height)*8), (0, 255, 0), 8)
else:
    print('none')

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()