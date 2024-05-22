import cv2
import numpy as np
import matplotlib as plt
#import keyboard

def checking_distance_between_objects(centroids, color_number, color):
    i = 1
    distance = np.empty((color_number, 2))
    while i < color_number:
        distance[i,0] =  abs(centroids[i,0] - centroids[i+1,0])
        distance[i,1] =  abs(centroids[i,1] - centroids[i+1,1])

        if color == 'red':
            farness = 23
        else:
            farness = 17
        if np.sqrt(distance[i,0]**2 + distance[i,1]**2) < farness:
            color_number = color_number - 1

        i = i + 1
    return color_number

# Function to make two skittles from one
'''
def deviding_objects(stats, number):
    nr = number
    nr_to_add = 0
    area = 200
    scale = 2.2
    for i in range(1, nr+1, 1):
        if stats[i,4] > area:
            if stats[i,2] > scale*min(stats[i,2], stats[i,3]) or stats[i,3] > scale*min(stats[i,2], stats[i,3]):
              nr_to_add = nr_to_add + 1
    return nr_to_add
'''


def Purple_function(img_hsv):
    # Purple, definitions of thresholds
    lower_purple = np.array([145, 50, 10])
    upper_purple = np.array([170, 255, 255])
    mask_purple = cv2.inRange(img_hsv, lower_purple, upper_purple)

    # Median filtring
    mask_purple = cv2.medianBlur(mask_purple, 7)
    # Define the kernel
    kernel = np.ones((4, 4), np.uint8)
    # Closing the image
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Counting each color
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_purple)
    # Subtracting one, because black area is also an area
    purple = num_labels - 1
    
    # If distance between two objects is less than a value, then it is one object
    purple = checking_distance_between_objects(centroids, purple)

    return purple, mask_purple

def Yellow_function(img_hsv):
    # Definitions of colours and masks
    # Yellow
    lower_yellow = np.array([19, 200, 120])
    upper_yellow = np.array([28, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # Median filtring
    mask_yellow = cv2.medianBlur(mask_yellow, 7)
    # Kernel
    kernel = np.ones((4, 4), np.uint8)
    # Opening
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel, iterations=1)
    # Counting each color
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_yellow)
    # Subtracting one, because black area is also an area
    yellow = num_labels - 1

    # If distance between two objects is less than a value, then it is one object
    yellow = checking_distance_between_objects(centroids, yellow)

    return yellow, mask_yellow

def Green_function(img_hsv):
    # Definitions of colours and masks
    # Yellow
    lower_green = np.array([31, 120, 123])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    # Median filtring
    mask_green = cv2.medianBlur(mask_green, 7)
    # Kernel
    kernel = np.ones((4, 4), np.uint8)
    # Opening
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    # Counting each color
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_green)
    # Subtracting one, because black area is also an area
    green = num_labels - 1

    # If distance between two objects is less than a value, then it is one object
    green = checking_distance_between_objects(centroids, green)

    return green, mask_green

def Red_function(img_hsv):
    # Definitions of colours and masks
    # Yellow
    Saturation = 140; Value  = 70
    lower_red = np.array([0, Saturation, Value])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)

    # Second mask
    lower_red_second = np.array([173, Saturation, Value])
    upper_red_second = np.array([180, 255, 255])
    mask_red_second = cv2.inRange(img_hsv, lower_red_second, upper_red_second)
    # Adding them together
    mask_red = mask_red_second #+ mask_red

    # Median filtring
    mask_red = cv2.medianBlur(mask_red, 7)
    # Kernel
    kernel = np.ones((3, 3), np.uint8)
    # Opening
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    # Counting each color
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_red)
    # Subtracting one, because black area is also an area
    red = num_labels - 1

    # If distance between two objects is less than a value, then it is one object
    red = checking_distance_between_objects(centroids, red, 'red')

    return red, mask_red



def detect(img):

    # Proper scaling of an image
    h, w, c = img.shape
    fx = 0.5; fy = 0.5
    
    if w < 800 or h < 800:
        fx = 1
        fy = 1

    if w > 800 or h > 800:
        fx = 0.6
        fy = 0.6

    if w> 1600 or h > 1600:
        fx = 0.2
        fy = 0.2

    img = cv2.resize(img, (0,0), fx = fx, fy = fy)

    # RGB to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, c = img.shape

    #yellow, mask_yellow = Yellow_function(img_hsv)
    #print(yellow)

    #green, mask_green = Green_function(img_hsv)
    #print(green)
    
    red, mask_red = Red_function(img_hsv)
    print(red)

    #purple, mask_purple = Purple_function(img_hsv)
    #print(purple)

    
    #cv2.imshow('purple mask', mask_purple)
    #cv2.imshow('green mask', mask_green)
    cv2.imshow('green mask', mask_red)
    #cv2.imshow('yellow mask', mask_yellow)
    cv2.imshow('image', img)
    

i = 0
while True:
    
    # Reading an image
    path = 'C:/Users/janlu/OneDrive/Pulpit/Janek/zadania studia/semestr 5/WdPO/WDPO_project/my_data/' + str(i) + '.jpg'
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    # Detect function
    detect(img)

    # 
    key =cv2.waitKeyEx(0)
    if (key == ord('d')):
        i = i + 1
        if i > 39: i=0
    elif(key == ord('a')):
        i = i-1
        if i < 0: i=39
    
