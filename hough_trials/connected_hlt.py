import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from hough_line_transform import hough_line_transform
from edge_detection import get_mnist_image, extract_edges, upscale_image

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def group_lines_by_angle(lines, angle_threshold=np.pi/144):  # 5 degrees threshold
    print("Number of lines detected: ", len(lines))
    grouped_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = calculate_angle(x1, y1, x2, y2)
        added = False
        for group in grouped_lines:
            if abs(group['angle'] - angle) < angle_threshold:
                group['lines'].append((x1, y1, x2, y2))
                added = True
                break
        if not added:
            grouped_lines.append({'angle': angle, 'lines': [(x1, y1, x2, y2)]})
    return grouped_lines

def find_endpoints(lines):
    x_coords = []
    y_coords = []
    for (x1, y1, x2, y2) in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def draw_grouped_lines(img, grouped_lines):
    print("Number of groups: ", len(grouped_lines))
    for group in grouped_lines:
        x1, y1, x2, y2 = find_endpoints(group['lines'])
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)  # Red color in BGR

def display_res(img, edges, lines):
    color_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    if lines is not None:
        grouped_lines = group_lines_by_angle(lines)
        draw_grouped_lines(color_edges, grouped_lines)
    
    # Display the images using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Source")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grouped Lines")
    plt.imshow(color_edges)
    plt.axis('off')
    plt.show()



(img, y) = get_mnist_image()
img = img.reshape((28, 28))
edges = extract_edges(img)

scale_factor = 10  # Increase the resolution by a factor of 2

# Bicubic Interpolation
upscaled_bicubic = upscale_image(edges, scale_factor, cv.INTER_CUBIC)
u = upscale_image(edges, scale_factor, cv.INTER_CUBIC)
lines = hough_line_transform(upscaled_bicubic)

display_res(u, upscaled_bicubic, lines)
