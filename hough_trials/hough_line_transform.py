import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage.exposure

from edge_detection import get_mnist_image, extract_edges
from edge_detection import visualize_img_edges
from edge_detection import upscale_image

def hough_line_transform(edges):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=1)  # Adjust parameters as needed
    return lines

def display_res(img, edges, lines):
    if lines is not None:
        print("Number of lines detected: ", len(lines))
    else:
        print("No lines detected.")
    color_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(color_edges, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)  # Red color in BGR

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
img = img.reshape((28, 28))
edges = extract_edges(img)

# Upscale the image using different interpolation methods
scale_factor = 1  # Increase the resolution by a factor of 2

# Bicubic Interpolation
upscaled_bicubic = upscale_image(edges, scale_factor, cv.INTER_CUBIC)

u = upscale_image(edges, scale_factor, cv.INTER_CUBIC)

lines = hough_line_transform(upscaled_bicubic)
display_res(u, upscaled_bicubic, lines)