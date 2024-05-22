import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  # open the default camera, indeks kamery w systemie
    key = ord('a') # kod asci przypisujemy do key, pojedyncze są równoważne z "
    while key!=ord('q'):
        #capture frame by frame
        ret, frame = cap.read() #zwracamy dwie rzeczy, czy sie udalo, i klatke obrazu
        #zatrzeskuje klatkę i pobiera

        #Our operation on the frame comes here
        #Convert RGB image to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Blur the image
        img_filtered = cv2.GaussianBlur(img_gray, (15,15), 5)
        #Detect edges on the blurred image
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)

        # Display the result of our processing
        cv2.imshow('original', frame)
        cv2.imshow('result', img_edges)
        cv2.imshow('result1', img_gray)
        cv2.imshow('result2', img_filtered)
        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(30) # bez tego nie odświeży się okno, ustalamy refresh rate
        # trzeba uwazac na zbyt mały refresh rate , lepiej dać większą, a czytać co którąś klatkę
        # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()
# i tak dwie ostatnie komendy się wykonają, nawet jak ich nie będzie, program kończy działanie


def Konrad():
    key = ord('a')
    while key != ord('q'):
        Konrad = cv2.imread("Konrad.jpg")
        cv2.imshow('Konrad 1', Konrad)

        #Konrad[0:500, 0:300] = Konrad[300:800,300:600]
        #cv2.imshow('Konrad 2',Konrad)

        pixel = Konrad[1000,1000]
        print(pixel)

        key = cv2.waitKey()
    print(np.shape(Konrad))
    cv2.destroyAllWindows()

def Split():
    key = ord('a')
    while key != ord('q'):
        cv2.imread("Color.png")
        

        key = cv2.waitKey()
    cv2.destroyAllWindows()




Konrad()
#main()