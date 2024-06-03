import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob2 as glob
import argparse
import os
import json


def preprocessing_img(input_dir,img,img_hsv,img_grey):

    # Sprawdzenie, czy podany katalog istnieje
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Podany katalog nie istnieje: {input_dir}")
    
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

def detecting_plate(grey_img):

    gray = grey_img.copy()
    
    # basic filtering and morphological operations
    kernel = np.ones((7,7),np.uint8)
    _, img_bin= cv2.threshold(gray, 157, 255, cv2.THRESH_BINARY)
    # TODO jeszcze mozna progowanie otsu sprobowac
    
    
    img_bin = cv2.medianBlur(img_bin, 5)
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # Counting each color
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)

    # Stwórz pusty obraz binarny o tych samych wymiarach co obraz etykiet
    binary_output = np.zeros(labels.shape, dtype=np.uint8)

    # Przejdź przez statystyki i znajdź komponenty spełniające warunki
    for i, stat in enumerate(stats):
        if (stat[0] >= 50 and stat[0] <= 700 and stat[1] >= 50  and stat[1] <= 500 and stat[2] <= 900 and stat[2] >= 350 and 
                                                stat[3] <= 300 and stat[3] >= 80 and stat[4] <= 90000 and stat[4] >= 30000):
            print("znalazlem!", stat)
            # Ustaw piksele odpowiadające znalezionemu komponentowi na 1 w obrazie binarnym
            binary_output[labels == i] = 255

    # morphed_image = cv2.morphologyEx(binary_output, cv2.MORPH_DILATE, kernel, iterations=10)
    morphed_image = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel,iterations=35)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# TODO wyprobowanie cv2.CHAIN_APPROX_TC89_L1 
    
    # Sprawdź, czy znaleziono jakiekolwiek kontury
    if len(contours) == 0:
        raise ValueError("Nie znaleziono żadnych konturów na obrazie.")
    
    # # Znajdź największy kontur
    largest_contour = max(contours, key=cv2.contourArea)
    
    # # Określ punkty graniczne (rogi) konturu
    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) != 4:
        raise ValueError("Nie znaleziono dokładnie czterech punktów granicznych.")
    
    # Wyodrębnij punkty graniczne
    boundary_points = approx.reshape((4, 2))
    
    # Narysuj punkty graniczne na obrazie 
    for point in boundary_points:
        cv2.circle(morphed_image, tuple(point), 10, (128, 128, 128), -1)  
    cv2.drawContours(morphed_image, [approx], -1, (150, 155, 150), 3)
    
    # TODO wyciagnac rogi i po kolei je ułożyć, operacja przeksztalcenia 
    # posortuj punkty graniczne od najmniejszej sumy pikseli do najwiekszej
    sorted_points = sorted([point.tolist() for point in boundary_points], key=lambda point: point[0] + point[1])
    sorted_points = np.float32(np.array(sorted_points))
    # print(type(sorted_points), sorted_points)
    
    # perspective transformation
    goal_points = np.float32([[0,0],[0,130],[700,0],[700,130]])
    
    M = cv2.getPerspectiveTransform(sorted_points, goal_points)
    
    dst = cv2.warpPerspective(gray,M,(700,130))
    
    # Wyświetlenie obrazu z zaznaczonymi punktami granicznymi
    cv2.imshow("Grey img", gray)
    # cv2.imshow("Boundary Points", morphed_image)
    cv2.imshow("perspective transform", dst)
    
    return dst



#TODO 
    
def main():

    parser = argparse.ArgumentParser(description="Przetwarzanie obrazów")
    parser.add_argument("input_dir", type=str, help="Ścieżka do katalogu ze zdjęciami.")
    args = parser.parse_args()


    key = None
    img = []; img_hsv = []; img_grey = []
    i = 0 
    preprocessing_img(args.input_dir, img, img_hsv, img_grey)

    while key != 27:
        if (key == ord('d')):
            i += 1
        elif key == ord('a'):
            i -= 1
        i = i%len(img)

        print(i, ":")
        plate = detecting_plate(img_grey[i])

        key = cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#         Testowany program zostanie uruchomiony z dwoma parametrami przekazanymi w linii poleceń: bezwzględną ścieżką do istniejącego folderu ze zdjęciami oraz bezwzględną ścieżką do nieistniejącego pliku wyjściowego.

# w jaki sposób skonfigurować kod pythona wewnątrz pliku, ktory ma zostac uruchomiony w powyższy sposób?