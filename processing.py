import cv2
import numpy as np
import matplotlib.pyplot as plt

from uitls import create_circular_kernel


def color_distance_segmentation(image, target_color):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    target_color_lab = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2Lab)[0][0]
    
    distance = np.linalg.norm(lab_image - target_color_lab, axis=2)    
    distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return distance_norm

#Temp
def compute_laplacian(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    cv2.imshow('Lab', laplacian)
    cv2.imshow('AbsLap', laplacian_abs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_path, target_color):
    image = cv2.imread(image_path)

    color_distance_image = color_distance_segmentation(image, target_color)
    ret, thresholded = cv2.threshold(color_distance_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Threshold: ", ret)


    kernel = create_circular_kernel(7)
    dialated  = cv2.dilate(thresholded,kernel,iterations = 9)
    errotion  = cv2.erode(dialated,kernel,iterations = 11)

    kernel = create_circular_kernel()
    dialated_2  = cv2.dilate(errotion,kernel,iterations = 1)


    cv2.imshow('Color Distance Segmentation', color_distance_image)
    cv2.imshow('Thresholded', thresholded)
    cv2.imshow('Dialated', dialated)
    cv2.imshow('Erosion', errotion)
    cv2.imshow('Dialated Second round', dialated_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    target_color = np.array([34,49,93], dtype=np.uint8)
    main('img.jpg', target_color)
    


