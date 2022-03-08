import numpy as np
from PIL import Image

#open image as np array
with Image.open('images/aef-CSN-III-3-1_088-600x900.jpg') as im:
    img = np.asarray(im)
    
# img = pil Image
# scale ration to scale by 
# file_name
def scale_image(img, scale):
    orginal_width = img.shape[0]
    orginal_height = img.shape[1]

    # calcualte the new size of image based on scale ratio
    new_width = int(orginal_width / scale)
    new_height = int(orginal_height / scale)
    scaled_img = np.zeros((new_width,new_height,3), dtype=np.uint8)

    # for each pixel in resized image find nearest pixel in orignal
    for x in range(new_width):
        for y in range(new_height):
            nx = int(x*scale)
            ny = int(y*scale)
            scaled_img[x,y] = img[nx,ny]

    # Convert numpy array back to PIL image format and save to disk
    return Image.fromarray(scaled_img)

def save_image(image, output_name):
    image.save(output_name + '.png', 'png')

# resize image with scale of 5
resized_image =scale_image(img, 5)
save_image(resized_image, 'resized_images/resized_x')