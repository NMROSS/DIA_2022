from pickle import FALSE
import numpy as np
from PIL import Image, ImageOps

def open_gray(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.grayscale(im))

#open image as np array
def open_image(path):
    with Image.open(path) as im:
        return np.asarray(im)

def save_image(image, output_name):
    image.save(output_name + '.png', 'png')

def convolution(image, kernal, stride,):
    kernal_size = len(kernal)
    x_size = int((image.shape[0] - kernal_size) / stride ) + 1
    y_size = int((image.shape[1] - kernal_size) / stride ) + 1
    output = np.zeros((x_size,y_size))

    for x in range(x_size):
        for y in range(y_size):
            output[x,y] = ( ((image[x:x+kernal_size, y:y+kernal_size]) * kernal ).sum())
        
    return output

def convolution_rgb(image, kernal, stride,):
    kernal_size = len(kernal)
    x_size = int((image.shape[0] - kernal_size) / stride ) + 1
    y_size = int((image.shape[1] - kernal_size) / stride ) + 1
    output = np.zeros((x_size,y_size, 3))

    for c in range(3):
        for x in range(x_size):
            for y in range(y_size):
                output[x,y,c] = ( ((image[x:x+kernal_size, y:y+kernal_size, c]) * kernal ).sum())
            
    return output

blur = np.array([
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1],
]) * (1/256)

sobel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1],
])

sobel_y = np.array([
        [-1,-2,-1],
        [0, 0, 0],
        [1,2,1],
])

def outline(image_path, save_name):
    img = open_gray(image_path)

    # apply both x and y kernal on image
    out_image_x = convolution(img, sobel_x, 1)
    out_image_y = convolution(img, sobel_y, 1)

    #combine x and y sobel to form on image of edges
    out_image = np.sqrt( np.square(out_image_x) + np.square(out_image_y) )

    out_image = Image.fromarray(np.uint8(out_image))
    save_image(out_image, save_name)


def blur_image(image_path, save_name):
    img = open_image(image_path)
    out_image = convolution_rgb(img, blur, 1)
    print(out_image)
    out_image = Image.fromarray(np.uint8(out_image))
    save_image(out_image, save_name)

outline('images/f53de51f6de846d6fff1b87922e7e3fb_3700x5500.jpg', 'outline1')
#blur_image('images/f53de51f6de846d6fff1b87922e7e3fb_3700x5500.jpg', 'blur4')

