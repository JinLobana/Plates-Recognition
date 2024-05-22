import numpy as np
import cv2

def empty_callback(value):
    pass

def main():

    lis2 = cv2.imread('2.jpg')
    dim = (1200,800)
    lis2 = cv2.resize(lis2, dim)
    name = "fox"
    name2 = "transformed fox"
    name3 = "transformed/original fox"
    cv2.namedWindow(name)
    cv2.namedWindow(name2)
    cv2.namedWindow(name3)

    full_picture3 = lis2.copy()
    quarter_1 = lis2[0:400, 0:600]
    quarter_2 = lis2[0:400, 600:1200]
    quarter_3 = lis2[400:800, 0:600]
    quarter_4 = lis2[400:800, 600:1200]

    quarter_1_grey = cv2.cvtColor(quarter_1, cv2.COLOR_BGR2GRAY)
    # Stack grayscale image along the third axis to match dimensions
    quarter_1_grey = np.stack([quarter_1_grey]*3, axis=-1)
    
    quarter_2_r = quarter_2
    quarter_2_r[:,:,0] = 0
    quarter_2_r[:,:,1] = 0
    #print(np.shape(quarter_2_r))

    quarter_3_g = quarter_3
    quarter_3_g[:,:,0] = 0
    quarter_3_g[:,:,2] = 0
    #print(np.shape(quarter_3_g))

    quarter_4_b = quarter_4
    quarter_4_b[:,:,1] = 0
    quarter_4_b[:,:,2] = 0

    upper_half = np.hstack((quarter_1_grey, quarter_2_r))
    lower_half = np.hstack((quarter_3_g, quarter_4_b))
    full_picture = np.vstack((upper_half, lower_half))
    full_picture2 = np.vstack((upper_half, lower_half))

    # create trackbars for color change
    cv2.createTrackbar('GREY', name2, 0, 255, empty_callback)
    cv2.createTrackbar('R', name2, 0, 255, empty_callback)
    cv2.createTrackbar('G', name2, 0, 255, empty_callback)
    cv2.createTrackbar('B', name2, 0, 255, empty_callback)

    #print(np.shape(full_picture2))

    while True:
        
        key_code = cv2.waitKey(10)
        if key_code ==27:
            break
        
        cv2.imshow(name, full_picture)
        cv2.imshow(name2, full_picture2)
        cv2.imshow(name3, full_picture3)
        
        grey = cv2.getTrackbarPos('GREY', name2)
        r = cv2.getTrackbarPos('R', name2)
        g = cv2.getTrackbarPos('G', name2)
        b = cv2.getTrackbarPos('B', name2)

        
        
        _, full_picture2[0:400, 0:600,:] = cv2.threshold(quarter_1_grey, grey, 255, cv2.THRESH_BINARY)
        _, full_picture2[0:400, 600:1200] = cv2.threshold(quarter_2_r, r, 255, cv2.THRESH_BINARY)
        _, full_picture2[400:800, 0:600] = cv2.threshold(quarter_3_g, g, 255, cv2.THRESH_BINARY)
        _, full_picture2[400:800, 600:1200] = cv2.threshold(quarter_4_b, b, 255, cv2.THRESH_BINARY)

        red_mask= full_picture2[0:400, 600:1200, 2]
        green_mask = full_picture2[400:800, 0:600, 1]
        blue_mask = full_picture2[400:800, 600:1200, 0]

        full_picture3[0:400, 600:1200, 2] = red_mask & full_picture3[0:400, 600:1200, 2]
        full_picture3[400:800, 0:600, 1] = green_mask & full_picture3[400:800, 0:600, 1] 
        full_picture3[400:800, 600:1200, 0] = blue_mask & full_picture3[400:800, 600:1200, 0]

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()