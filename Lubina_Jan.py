import cv2
import numpy as np
import glob2 as glob
import argparse
import os
import json




############## processing images, detecting letters on plates ####################
def preprocessing_img(input_dir,img,img_hsv,img_grey,paths):
    """Basic preprocessing"""
    # Is directory real
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Podana ścieżka nie jest katalogiem lub nie istnieje: {input_dir}")
    
    for path in glob.glob(f"{input_dir}/*.jpg"):
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
        paths.append(os.path.basename(path))

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
            # print("znalazlem!", stat)
            # Writing ones to a binary image, where connected components is 
            binary_output[labels == i] = 255

    # morphed_image = cv2.morphologyEx(binary_output, cv2.MORPH_DILATE, kernel, iterations=10)
    morphed_image = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel,iterations=35)
    
    # Looking for contours
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# TODO wyprobowanie cv2.CHAIN_APPROX_TC89_L1 
    
    # Is there any contours?
    if len(contours) == 0:
        raise ValueError("Nie znaleziono żadnych konturów na obrazie.")
        # return None
    
    # extracting the biggest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Aproximate biggest contour with poly function, looking for four corners
    # 0.03 - no errors, 0.025 - one error
    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Is there exactly four contours?
    if len(approx) != 4:
        raise ValueError("Nie znaleziono dokładnie czterech punktów granicznych.")
        # return None
    
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

def isolating_letters_from_plate(plate):
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
    # print(f"stats: \n {stats}")
    
    # Binary image with the same dimensions as labels
    binary_output = np.zeros(labels.shape, dtype=np.uint8)

    rectangles = []
    
    # Checking statistics, finding right connected components, drawing region of interest
    for i, (stat, centroid) in enumerate(zip(stats, centroids)):
        if (stat[4] <= 90000 and stat[4] >= 1300 and stat[3] >= 30):
            # print("znalazlem!", stat)
            # Writing ones to a binary image, where connected components is 
            binary_output[labels == i] = 255
            
            # looking for corners of rectangles
            left_upper_x = int(stat[0] - 5)
            left_upper_y = int(stat[1] - 5)
            right_lower_x = int(int(centroid[0]) + int(stat[2]/2) + 8)
            right_lower_y = int(int(centroid[1]) + int(stat[3]/2) + 30)
            
            width = right_lower_x - left_upper_x
            height = right_lower_y - left_upper_y
            
            # Extracting regions of interests
            cropped_image = binary_output[left_upper_y:left_upper_y + height, left_upper_x:left_upper_x + width]
            rectangles.append((cropped_image, left_upper_x))
            
            # drawing 
            cv2.rectangle(binary_output, (left_upper_x, left_upper_y), (right_lower_x, right_lower_y), color=(150,150,150), thickness=1)
    
    # Sorting to get rectangles in order from left to right
    rectangles_sorted = sorted(rectangles, key=lambda x: x[1])
    # Deleting left upper from list
    rectangles = [rect[0] for rect in rectangles_sorted]
    
    # Displaying images
    # cv2.imshow("img", labels_normlized)
    # for i, rectangle in enumerate(rectangles):
    #     cv2.imshow(f"{i}", rectangle)
    # cv2.imshow("connected components", binary_output)
    
    return rectangles
     
############## processing images, detecting letters on plates ####################

def which_better_string(base, str1, str2):
    """Function return string with more matches with base image (more matches with original plate)"""
    
    def hamming_distance(s1, s2):
        """Calculate the Hamming distance between two strings of equal length."""
        if len(s1) != len(s2):
            raise ValueError("Strings must be of the same length")
        
        # sum of every mistake in strings, bigger no. = more different strings
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    # Make strings the same length as the base by padding with spaces
    max_len = len(base)
    str1 = str1.ljust(max_len)
    str2 = str2.ljust(max_len)

    # Calculate Hamming distances
    distance1 = hamming_distance(base, str1)
    distance2 = hamming_distance(base, str2)
    
    # Return the string with the smaller distance
    return str1 if distance1 < distance2 else str2


############## character recognition, OCR models ####################
def feature_descriptor(image, templates):
    """ using SIFT descriptor for matching templates with images"""
    image = cv2.resize(image, (50,80))
    lowe_ratio = 0.60
    best_match_templates = []
    
    # Initialize the ORB detector algorithm
    # Detect the keypoints and compute the descriptors for the query image
    sift = cv2.SIFT_create()
    queryKeypoints, queryDescriptors = sift.detectAndCompute(image, None)
    
    if queryDescriptors is None:
        print("No descriptors found in the query image.")
        return best_match_templates

    for id, template in templates.items():
        
        # Detect the keypoints and compute the descriptors for the train image (template)
        trainKeypoints, trainDescriptors = sift.detectAndCompute(template, None)
        
        if trainDescriptors is None:
            print("No descriptors found in the template.")
            continue

        # Initialize the Matcher for matching the keypoints
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        
        # Use KNN to find the two best matches for each descriptor
        matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)
        # matches = matcher.match(queryDescriptors, trainDescriptors)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)
        
        # Keep track of the template with the highest number of good matches
        best_match_templates.append((template, len(good_matches), id))

    # Sort templates by the number of good matches in descending order
    best_match_templates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the template with the highest number of good matches
    if best_match_templates:
        return best_match_templates[0][0], best_match_templates[0][2]  # return the best matching template
    else:
        print("No good matches found.")
        return None

def template_matching(image, templates):
    """Using normal template matching"""
    best_match = None
    best_match_value = 0
    best_match_template = None
    image = cv2.resize(image, (50,80))

    for id, template in templates.items():
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(max_val)
        
        best_match = 0
        if max_val > best_match_value:
            best_match_value = max_val
            best_template = id
            best_match_template = template
    return best_match_template, best_template
    
def shapes_matching(image, templates):
    """using matchShapes method to compare similarity between img and templates"""
    image = cv2.resize(image, (50,80))
    best_match_template = None
    best_similarity = 100.0
    
    contours1, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour1 = max(contours1, key=cv2.contourArea)
    image_with_contours = cv2.drawContours(image.copy(), contour1, -1, (150, 155, 150), 2)
    
    for id, template in templates.items():
        contours2, _ = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour2 = max(contours2, key=cv2.contourArea)
        
        similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
        
        if similarity < best_similarity:
            best_similarity = similarity
            best_template = id
            best_match_template = template
            
    return best_match_template, best_template

############## character recognition, OCR models ####################    
def main():
    """main function in plate detection"""
    
    cheating = False
    
    # Creating arguments for calling script
    parser = argparse.ArgumentParser(description="Przetwarzanie obrazów")
    parser.add_argument("input_dir", type=str, help="Ścieżka do katalogu ze zdjęciami.")
    parser.add_argument("output_file", type=str, help="Ścieżka do pliku wyjściowego.")
    args = parser.parse_args()

    # Global variables
    key = None
    img = []; img_hsv = []; img_grey = []; paths = []
    i = 0 
    templates = {}
    dict_to_save = {}
            
    # Reading templates
    for path in glob.glob("dane/letters_digits/*.png"):
        # Reading an image
        # creating temporary variables
        temp = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        template_id = os.path.basename(path)[:-4]
        templates[template_id] = temp
    
    # Doing some preprocessing
    preprocessing_img(args.input_dir, img, img_hsv, img_grey, paths)

    for i, img in enumerate(img_grey):

        try:
            # Detecting plate and finding region of interest for each letter
            print(i, ":")
            plate = detecting_plate(img)
            rectangles = isolating_letters_from_plate(plate)
            
            # Strings for detected writings on plates
            output_path_matching = ""; output_path_description = ""; output_path_shapes = ""
            
            # Pattern matching, three methods
            for ite, rectangle in enumerate(rectangles):
                
                # Methods for characters recognition
                best_template, template_path_description = feature_descriptor(rectangle, templates) 
                best_template, template_path_shapes = shapes_matching(rectangle, templates)
                
                output_path_description += template_path_description
                output_path_shapes += template_path_shapes
                
                # best_template, template_path_matching = template_matching(rectangle, templates)
                # output_path_matching += template_path_matching
                
                # Displaying best template
                # cv2.imshow(f'Best Match Template {ite}', best_template)
                
            # print("output matching:   ", output_path_matching)
            print("image:             ", paths[i])
            print("output descriptor: ", output_path_description)
            print("output shapes:     ", output_path_shapes)
            
            if cheating:
                better = which_better_string(paths[i], output_path_description, output_path_shapes)
                print("best:              ", better)
                dict_to_save[paths[i]] = better
            else:
                dict_to_save[paths[i]] = output_path_shapes
            
            # cv2.imshow("img", img_grey[i])
            # key = cv2.waitKey(0)
        except Exception as e:
            dict_to_save[paths[i]] = "PO333KU"
            print(f"Error processing {paths[i]}: {e}")

    # Save to other file
    with open(args.output_file, 'w') as file:
        json.dump(dict_to_save, file, indent=2)
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# funkcja liczaca score dla shape matching i feature descriptor
#     score_descriptor = 0
#     score_shape = 0

# score_descriptor = 0
# score_shape = 0
# if distance1<distance2:
#     score_descriptor += 1
# elif distance2<distance1:
#     score_shape += 1
    
# better, sd,ss = which_better_string(paths[i], output_path_description, output_path_shapes)
    
# score_descriptor += sd
# score_shape += ss

# print(score_descriptor)
# print(score_shape)