import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from edge_detection import get_mnist_image, extract_edges
from edge_detection import upscale_image

def hough_circle_transform(edges):
    gray = cv.medianBlur(edges, 5)
    gray = edges
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
    return circles
    
def display_res(img, edges, circles):
    if circles is not None:
        print("Number of circles detected: ", len(circles))
    else:
        print("No circles detected.")
    color_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(color_edges, center, radius, (255, 0, 255), 3)
    
    
    # Display the images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Source")
    #print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    #print(edges.shape)
    plt.title("Detected Lines (in red) - Probabilistic Hough Line Transform")
    plt.imshow(color_edges)
    #plt.imshow(cv.cvtColor(color_edges, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with plt
    plt.axis('off')

    plt.show()

(img, y) = get_mnist_image()
edges = extract_edges(img)

scale_factor = 10
upscaled_bicubic = upscale_image(edges, scale_factor, cv.INTER_CUBIC)
u = upscale_image(edges, scale_factor, cv.INTER_CUBIC)

circles = hough_circle_transform(edges)
display_res(u, upscaled_bicubic, circles)