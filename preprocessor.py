import matplotlib.pyplot as plt
import glob
from PIL import Image

# Grabs the names of all the png files at the specified path
image_names = glob.glob(filepath + "/*.png")

for file in image_names:
    image = Image.open(file, 'r')
    image = image.convert('L') #converts to grayscale
    image = whitethresh(image)
    image = blackthresh(image)
    image.save(file+'modified.png', "PNG")

def whitethresh(image):
    # iterates over each pixel to create a mask. if the value > 240, it sets it to 255 in the mask
    mask = image.point(lambda pixel: pixel>=240 and 255)

    #overlays the mask wherever the mask was valid
    image.paste(mask, None, mask)
    return image

def blackthresh(image):
    mask = image.point(lambda pixel: pixel<240 and int(pixel*254/239))

    #overlays the mask wherever the mask was valid
    image.paste(mask, None, mask)
    return image
