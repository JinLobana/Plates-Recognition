import numpy as np
import cv2

def main():

    dok = "dokument"
    dok2 = "dokument2"
    cv2.namedWindow(dok, cv2.WINDOW_NORMAL)
    cv2.moveWindow(dok, 0, 0)
    cv2.resizeWindow(dok, 1920, 1080)
    #cv2.namedWindow(dok2, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(dok2, 1920, 1080)
    docs = cv2.imread("/home/jinlobana/Desktop/Jjs/studia/SW/dokument.jpg")
    kernel = np.ones((3,3),np.uint8)

    docs_gray = cv2.cvtColor(docs, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(docs_gray, 50, 150)
    # # Find contours in the edge-detected image
    # contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    docs_median = cv2.medianBlur(docs_gray,5) #wyglada gorzej
    _, docs_threshold = cv2.threshold(docs_gray, 128, 255, cv2.THRESH_BINARY)
    
    docs_opened = cv2.morphologyEx(docs_threshold, cv2.MORPH_OPEN, kernel)
    
    #docs_adaptive = cv2.adaptiveThreshold(docs_opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1) # nie daje lepszych rezultatów

    

    cv2.imwrite("dokument_przetworzony.jpg", docs_opened)

    




    while True:
        
        key_code = cv2.waitKey(10)
        if key_code ==27:
            break

        cv2.imshow(dok, docs_opened)
#        cv2.imshow(dok2, docs_adaptive)



if __name__ == '__main__':
    main()