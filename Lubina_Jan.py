import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob2 as glob


def preprocessing_img(img,img_hsv,img_grey):
    
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

    _, img_bin= cv2.threshold(grey_img, 157, 255, cv2.THRESH_BINARY)
    img_bin = cv2.medianBlur(img_bin, 5)

    # # Define the kernel
    # kernel = np.ones((4, 4), np.uint8)
    # # Closing the image
    # mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Counting each color
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)

    # Stwórz pusty obraz binarny o tych samych wymiarach co obraz etykiet
    binary_output = np.zeros(labels.shape, dtype=np.uint8)

    # Przejdź przez statystyki i znajdź komponenty spełniające warunki
    for i, stat in enumerate(stats):
        if (stat[0] >= 50 and stat[0] <= 700 and stat[1] >= 50  and stat[1] <= 500 and stat[2] <= 900 and stat[2] >= 350 and 
                                                stat[3] <= 300 and stat[3] >= 80 and stat[4] <= 90000 and stat[4] >= 30000):
            print("znalazlem!", stat)
            # Ustaw piksele odpowiadające znalezionemu komponentowi na 1 w obrazie binarnym
            binary_output[labels == i] = 1

    # Wyświetlenie obrazu binarnego przy użyciu OpenCV
    cv2.imshow("Binary Output", binary_output * 255) 

    # Normalizuj obraz etykiet, aby przeskalować wartości do zakresu 0-255 (dla obrazu szarości)
    # labels_normalized = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # # Alternatywnie, przekonwertuj etykiety na obraz kolorowy (do wyświetlania w kolorze)
    # label_colormap = cv2.applyColorMap(labels_normalized, cv2.COLORMAP_JET)

    # # Wyświetlenie obrazu etykiet przy użyciu OpenCV
    # cv2.imshow("Labels (grayscale)", labels_normalized)
    # cv2.imshow("Labels (colormap)", label_colormap)

    return img_bin

    
def main():
    key = None
    img = []; img_hsv = []; img_grey = []
    i = 0 
    preprocessing_img(img,img_hsv,img_grey)

    while key != 27:
        if (key == ord('d')):
            i += 1
        elif key == ord('a'):
            i -= 1
        i = i%len(img)

        print(i, ":")
        contour = detecting_plate(img_grey[i])
        
        cv2.imshow("img", contour)

        key = cv2.waitKey(0)

    cv2.destroyAllWindows()

main()

#         Testowany program zostanie uruchomiony z dwoma parametrami przekazanymi w linii poleceń: bezwzględną ścieżką do istniejącego folderu ze zdjęciami oraz bezwzględną ścieżką do nieistniejącego pliku wyjściowego.

# w jaki sposób skonfigurować kod pythona wewnątrz pliku, ktory ma zostac uruchomiony w powyższy sposób?