import numpy as np
import cv2
import math
import imutils

def empty_callback(value):
    pass

def main():

    lis = cv2.imread('2.jpg')
    dim = (1200,800)
    lis = cv2.resize(lis, dim)
    name = "fox"
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)

    h, w= lis.shape[:2]
    center = (w/2, h/2)
    

    cv2.createTrackbar('angle', name, 0, 360, empty_callback)



    while True:
        
        key_code = cv2.waitKey(10)
        if key_code ==27:
            break

        angle = cv2.getTrackbarPos('angle', name)
        nW = int((h * math.sin(angle)) + (w * math.cos(angle)))
        nH = int((h * math.cos(angle)) + (w * math.sin(angle)))

        
       # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1) 

        #rotate_matrix[0, 2] += (nW / 2) - center[0]
        #rotate_matrix[1, 2] += (nH / 2) - center[1]

        #rotated_image = cv2.warpAffine(lis, rotate_matrix, lis.shape[1::-1], flags=cv2.INTER_LINEAR)
        #rotated_image = cv2.warpAffine(src=lis, M=rotate_matrix, dsize=(nW, nH)) 

        rotated_image = imutils.rotate_bound(lis, angle)
        #cv2.imshow("Rotated Without Cropping", rotated_image)


        #cv2.resizeWindow(name, int(w + math.cos(angle)*h), int(h + math.sin(angle)*w))
        #rotated_image = cv2.resize(rotated_image, (int(w + math.cos(angle)), int(h + math.sin(angle))))
        cv2.imshow(name, rotated_image)
        



if __name__ == '__main__':
    main()