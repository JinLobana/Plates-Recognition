import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def zad_1():

    cube = cv2.imread('C:/Users/janlu/OneDrive/Pulpit/Janek/zadania studia/semestr 5/WdPO/qr.jpg')
    t1_start = time.perf_counter_ns()
    cube_inter_linear = cv2.resize(cube, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LINEAR)
    t1_stop = time.perf_counter_ns()

    t2_start = time.perf_counter_ns()
    cube_inter_nearest = cv2.resize(cube, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_NEAREST)
    t2_stop = time.perf_counter_ns()

    t3_start = time.perf_counter_ns()
    cube_inter_area = cv2.resize(cube, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_AREA)
    t3_stop = time.perf_counter_ns()

    t4_start = time.perf_counter_ns()
    cube_inter_lanczos4 = cv2.resize(cube, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LANCZOS4)
    t4_stop = time.perf_counter_ns()

    print(f"elapsed time linear: {(t1_stop-t1_start)/1000}\n elapsed time neares: {(t2_stop - t2_start)/1000}\n elapsed time area: {(t3_stop - t3_start)/1000}\n elapsed time lanczos: {(t3_stop - t3_start)/1000} ")

    plt.imshow(cube)

    '''
    figure, axis = plt.subplots(2,2)
    axis[0, 0].imshow(cube_inter_linear)
    axis[0, 0].set_title("linear")
    axis[0, 1].imshow(cube_inter_nearest)
    axis[0, 1].set_title("nearest")
    axis[1, 0].imshow(cube_inter_area)
    axis[1, 0].set_title("area")
    axis[1, 1].imshow(cube_inter_lanczos4)
    axis[1, 1].set_title("lanczos 4")
    '''
    plt.figure()
    plt.imshow(cube_inter_linear)
    plt.figure()
    plt.imshow(cube_inter_nearest)
    plt.figure()
    plt.imshow(cube_inter_area)
    plt.figure()
    plt.imshow(cube_inter_lanczos4)

    plt.show()

def zad_2():
    color = cv2.imread("C:/Users/janlu/OneDrive/Pulpit/Janek/zadania studia/semestr 5/WdPO/Color.png")
    x = 600; y = 600
    value = np.uint8([255])
    color_resized = cv2.resize(color, (x,y), interpolation=cv2.INTER_LINEAR)
    rows, columns, bgr = color_resized.shape
    print(rows,columns, bgr)

    color_negative = np.zeros(color_resized.shape, dtype=np.uint8)

    for row in range(rows):
        for column in range(columns):
            for canal in range(bgr):
                color_negative[row, column, canal] = 255 - color_resized[row, column, canal]

    cv2.imshow("image", color_resized)
    cv2.imshow("image 2", color_negative)
    while(1):
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
    



zad_2()
