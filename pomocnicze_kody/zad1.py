import numpy as np
import cv2

def main():
    
    lis1 = cv2.imread('1.jpg')
    dim = (600, 400)
    lis1 = cv2.resize(lis1, dim)
    lis2 = cv2.imread('2.jpg')
    lis2 = cv2.resize(lis2, dim)
    lis3 = cv2.imread('3.jpg')
    lis3 = cv2.resize(lis3, dim)

    i = 0
    j = 0
    time = 1000
    obraz = lis1
    slideshow = False

    print(ord('a'))

    while True:
        
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break
        
        cv2.imshow("lis", obraz)

        if key_code == ord('a'):
            slideshow = not slideshow
        if slideshow == True:
            
            if i == 4: i = 1
            if i == 1: obraz = lis1
            if i == 2: obraz = lis2
            if i == 3: obraz = lis3

            #print(i)
            j = j + 10 
            if j == time:
                j = 0
                i = i + 1

        print(time)
        if key_code == ord('z'):
            time = time - 100
        if key_code == ord('x'):
            time = time + 100
                
        

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()