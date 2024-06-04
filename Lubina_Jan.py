import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob2 as glob
import argparse
import os
import json


def preprocessing_img(input_dir,img,img_hsv,img_grey ,paths):
    """Basic preprocessing"""
    # Is directory real
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Podana ścieżka nie jest katalogiem lub nie istnieje: {input_dir}")
    
    for path in glob.glob("dane/train_1/*.jpg"):
        # Reading an image
        # creating temporary variables
        temp = cv2.imread(path,cv2.IMREAD_COLOR)
        temp = cv2.resize(temp, (1200,800))
        temp_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        temp_grey = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        # appending read images to lists
        img.append(temp)
        img_hsv.append(temp_hsv)
        img_grey.append(temp_grey)
        
        # appending paths
        paths.append(path)

def detecting_plate(grey_img):
    """finding plate on an image, extracting it"""
    gray = grey_img.copy()
    
    # basic filtering and morphological operations
    kernel = np.ones((7,7),np.uint8)
    _, img_bin= cv2.threshold(gray, 157, 255, cv2.THRESH_BINARY)
    # TODO jeszcze mozna progowanie otsu sprobowac
    img_bin = cv2.medianBlur(img_bin, 5)
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # creating list of connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)

    # Binary image with the same dimensions as labels
    binary_output = np.zeros(labels.shape, dtype=np.uint8)

    # Checking statistics, finding right connected components
    for i, stat in enumerate(stats):
        if (stat[0] >= 50 and stat[0] <= 700 and stat[1] >= 50  and stat[1] <= 500 and stat[2] <= 900 and stat[2] >= 350 and 
                                                stat[3] <= 300 and stat[3] >= 80 and stat[4] <= 90000 and stat[4] >= 30000):
            print("znalazlem!", stat)
            # Writing ones to a binary image, where connected components is 
            binary_output[labels == i] = 255

    # morphed_image = cv2.morphologyEx(binary_output, cv2.MORPH_DILATE, kernel, iterations=10)
    morphed_image = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel,iterations=35)
    
    # Looking for contours
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# TODO wyprobowanie cv2.CHAIN_APPROX_TC89_L1 
    
    # Is there any contours?
    if len(contours) == 0:
        raise ValueError("Nie znaleziono żadnych konturów na obrazie.")
    
    # extracting the biggest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Aproximate biggest contour with poly function, looking for four corners
    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Is there exactly four contours?
    if len(approx) != 4:
        raise ValueError("Nie znaleziono dokładnie czterech punktów granicznych.")
    
    # Reshaping corners
    boundary_points = approx.reshape((4, 2))
    
    # Drawing contour and corners
    for point in boundary_points:
        cv2.circle(morphed_image, tuple(point), 10, (128, 128, 128), -1)  
    cv2.drawContours(morphed_image, [approx], -1, (150, 155, 150), 3)
    
    # Sorting corners from the lowest sum of pixels (x + y) 
    sorted_points = sorted([point.tolist() for point in boundary_points], key=lambda point: point[0] + point[1])
    sorted_points = np.float32(np.array(sorted_points))
    # print(type(sorted_points), sorted_points)
    
    # Perspective transformation
    goal_points = np.float32([[0,0],[0,130],[700,0],[700,130]])
    
    M = cv2.getPerspectiveTransform(sorted_points, goal_points)
    
    dst = cv2.warpPerspective(gray,M,(700,130))
    
    # Showing images
    # cv2.imshow("Grey img", gray)
    # cv2.imshow("Boundary Points", morphed_image)
    # cv2.imshow("perspective transform", dst)
    
    return dst

def plate_segmentation(plate):
    """finding region of interest for every letter/number"""
    plate_cp = plate.copy()
    
    # morphological transfomations
    kernel = np.ones((5,5),np.uint8)
    _, img_bin= cv2.threshold(plate_cp, 130, 255, cv2.THRESH_BINARY)
    img_bin = cv2.medianBlur(img_bin, 5)
    closing = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    
    # contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # for idx in range(int(len(contours))):
    #     cv2.drawContours(closing, contours, idx, (150, 155, 150), 3)
        
    # add rounding box of white, for better character fragmentation    
    img = np.ones((230, 800), dtype=np.uint8)*255
    x_offset=y_offset=50
    img[y_offset:y_offset+closing.shape[0], x_offset:x_offset+closing.shape[1]] = closing
    
    # making negative for binary image, for connected components function to work correctly
    img_inverted = cv2.bitwise_not(img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_inverted)
    
    # normalizing labels for gray scale labels
    labels_normlized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(f"stats: \n {stats}")
    
    # Binary image with the same dimensions as labels
    binary_output = np.zeros(labels.shape, dtype=np.uint8)

    # Checking statistics, finding right connected components
    for i, stat in enumerate(stats):
        if (stat[0] >= 50 and stat[0] <= 700 and stat[1] >= 50  and stat[1] <= 500 and stat[2] <= 900 and stat[2] >= 350 and 
                                                stat[3] <= 300 and stat[3] >= 80 and stat[4] <= 90000 and stat[4] >= 30000):
            print("znalazlem!", stat)
            # Writing ones to a binary image, where connected components is 
            binary_output[labels == i] = 255
    
    # Displaying images
    cv2.imshow("img", labels_normlized)
    cv2.imshow("connected components", binary_output)
    

# 14 | 54 | 14 | 54 | 40 | 43 | 14 | 43 | 14 | 43 | 14 | 43 | 14 | 43 | 14
#TODO plate segmentation, template matching
    
def main():

    parser = argparse.ArgumentParser(description="Przetwarzanie obrazów")
    parser.add_argument("input_dir", type=str, help="Ścieżka do katalogu ze zdjęciami.")
    args = parser.parse_args()


    key = None
    img = []; img_hsv = []; img_grey = []; paths = []
    i = 0 
    preprocessing_img(args.input_dir, img, img_hsv, img_grey, paths)

    while key != 27:
        if (key == ord('d')):
            i += 1
        elif key == ord('a'):
            i -= 1
        i = i%len(img)

        print(i, ":")
        plate = detecting_plate(img_grey[i])
        
        plate_segmentation(plate)

        key = cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#         Testowany program zostanie uruchomiony z dwoma parametrami przekazanymi w linii poleceń: bezwzględną ścieżką do istniejącego folderu ze zdjęciami oraz bezwzględną ścieżką do nieistniejącego pliku wyjściowego.

# w jaki sposób skonfigurować kod pythona wewnątrz pliku, ktory ma zostac uruchomiony w powyższy sposób?