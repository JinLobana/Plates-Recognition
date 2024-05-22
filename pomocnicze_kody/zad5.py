import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def modul_kierunek_gradientu():
    img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1200,800))
    assert img is not None, "file could not be read, check with os.path.exists()"

    Prewitt_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    Sobel_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    Prewitt_img = cv2.filter2D(src=img, ddepth=-1, kernel=Prewitt_kernel) 
    Sobel_img = cv2.filter2D(src=img, ddepth=-1, kernel=Sobel_kernel) 

    while True:
        
        cv2.imshow("Prewitt", Prewitt_img)
        cv2.imshow("Sobel", Sobel_img)
        
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

    cv2.destroyAllWindows()
    


def krawedzie_Canny():
    img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv2.Canny(img,39,45)
    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def shapes():
    img = cv2.imread(cv2.samples.findFile('shapes.jpg'))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=150)
    
    circles = np.uint16(np.around(circles))
    for circle in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(circle[0], circle[1]), circle[2], (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(img,(circle[0], circle[1]), 2, (0,0,255), 3)
    
    while True:
    
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

        cv2.imshow('houghlines3.jpg',img)
    cv2.destroyAllWindows()



#shapes()
#modul_kierunek_gradientu()
krawedzie_Canny()
    




    
    
    

