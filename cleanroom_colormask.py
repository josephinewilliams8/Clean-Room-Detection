import cv2
import numpy as np

img = cv2.imread('cleanroom_pics/cleanroom2.jpg')

#converting to an rgb image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('cleanroom', rgb)
cv2.waitKey(0)

# declaring bounds to find color mask
teal_low = np.array([38,45,0], dtype=np.int64)
teal_high = np.array([90, 140, 20], dtype=np.int64)

mask = cv2.inRange(rgb, teal_low, teal_high)
countour, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

# show the contoured image
cv2.imshow('teal mask', mask)
cv2.waitKey(0)