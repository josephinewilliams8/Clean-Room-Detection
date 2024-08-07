import cv2
import numpy as np

img = cv2.imread('cleanroom2.jpg')

#converting to an hsv image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('cleanroom', hsv)
cv2.waitKey(0)

# declaring bounds to find color mask
blue_low = np.array([0,0,0], dtype=np.int64)
blue_high = np.array([23, 20, 5], dtype=np.int64)

mask = cv2.inRange(hsv, blue_low, blue_high)
countour, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)

# show the contoured image
cv2.imshow('blue mask', mask)
cv2.waitKey(0)