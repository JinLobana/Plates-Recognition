import cv2
import numpy as np
import matplotlib.pyplot as plt



def empty_callback(value):
    print(f'Trackbar reporting for duty with value: {value}')

cv2.namedWindow('image')
cv2.createTrackbar('TH', 'image', 0, 255, empty_callback)
cv2.createTrackbar('type', 'image', 0, 4, empty_callback)

def choose_thresh(i):
    if i == 0: return cv2.THRESH_BINARY
    if i == 1: return cv2.THRESH_BINARY_INV
    if i == 2: return cv2.THRESH_TRUNC
    if i == 3: return cv2.THRESH_TOZERO
    if i == 4: return cv2.THRESH_TOZERO_INV


# create switch for ON/OFF functionality
cv2.createTrackbar('0 : OFF \n1 : ON', 'image', 1, 1, empty_callback)

Konrad = cv2.imread("C:/Users/janlu/OneDrive/Pulpit/Janek/zadania studia/semestr 5/WdPO/Konrad.jpg")
Konrad_gray = cv2.cvtColor(Konrad, cv2.COLOR_BGR2GRAY)

Konrad_resized = cv2.resize(Konrad_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
img_current = Konrad_resized
while True:
    # get current positions of four trackbars
    th = cv2.getTrackbarPos('TH', 'image')
    s = cv2.getTrackbarPos('0 : OFF \n1 : ON', 'image')
    type = cv2.getTrackbarPos('type', 'image')

    if s == 0:
        #assign zeros to all pixels
        img_current[:] = 0

    else:
        # assign the same BGR color to all pixels
        ret, img_current = cv2.threshold(Konrad_resized, th, 255, choose_thresh(type))

    cv2.imshow('image', img_current)

    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()