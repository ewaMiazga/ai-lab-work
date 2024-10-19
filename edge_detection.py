import cv2 as cv
import numpy as np
from chainer import datasets
from PIL import Image
import matplotlib.pyplot as plt

def visualize_one_image(data, label, shape):
    plt.figure()
    plt.imshow(data.reshape(shape), cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def resize_images(images, new_size):
    resized_images = []
    for img in images:
        pil_img = Image.fromarray(img.reshape(28, 28))
        resized_img = pil_img.resize(new_size, Image.LANCZOS)
        resized_images.append(np.array(resized_img).reshape(-1))
    return np.array(resized_images)

def get_mnist_image():
    train, test = datasets.get_mnist()  
    #x_train_resized = resize_images(train._datasets[0], (14, 14))
    x_train = np.array(train._datasets[0])
    
    inx = np.random.randint(0, x_train.shape[0])
    return x_train[inx], train._datasets[1][inx]

def extract_edges(img, shape):
    img = img.reshape(shape)
    img = (img * 255).astype(np.uint8)

    _, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel, iterations=2)
    #edges = cv.Canny(img, 255/3, 255, 3)
    edges = cv.Canny(binary_img, 100, 200)
    return edges

def extract_contour(img, shape):
    # Reshape the image to the specified shape
    img = img.reshape(shape)
    
    # Convert the image to uint8 type
    img = (img * 255).astype(np.uint8)
    
    # Find contours in the image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw the contours
    edges = np.zeros_like(img)
    
    # Draw the contours on the empty image
    cv.drawContours(edges, contours, -1, (255), 1)
    
    return edges

def visualize_img_edges(img, y, edges, shape):
    plt.subplot(121),plt.imshow(img.reshape(shape),cmap = 'gray')
    plt.title('Original Image ' + str(y)), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image ' + str(y)), plt.xticks([]), plt.yticks([])
    plt.show()
    return 

def upscale_image(image, scale_factor, interpolation):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    return cv.resize(image, dim, interpolation=interpolation)

#(img, y) = get_mnist_image()

#edges = extract_edges(img)
#visualize_img_edges(img, y, edges)