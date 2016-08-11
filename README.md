# Sketch-Cleaner

## Overview and Primary Considerations

This project aims to train a neural network that can take a rough sketch and output a cleaned lineart. The output doesn’t have to be perfect, but should be good enough to avoid the bulk of the inking and lining process. To accomplish this, there’s several parts that needs to be completed:

1. An image preprocessor

2. A bootstrapper

3. A scoring function

4. A sketch generator

5. The neural network

The overall process will work like this. 

We will first need to acquire thousands of line art images from the internet, possibly up to hundreds of thousands. Then we will batch feed the images through the preprocessor to do some initial cleaning, resizing, and saving the images in a common file format. The bootstrapper will attempt to generate more data from a smaller number of images to improve sample size. The scoring function is an attempt to give a score on the similarity between two pieces of lineart. Then, the sketch generator will create sketches for each piece of lineart, possibly more than one. This sketch should meet a some minimum score on the similarity to the original image. Finally the neural network will take these generated sketches and attempt to recreate the original image. Again, the scoring function will be used here to update the neural network. Finally, we will validate the neural network on the validation dataset. Check the related sections for more information.

We expect to need a minimum of 10,000 images to get any results and possibly up to 100,000 to fully train the neural network due to the complexity of the task. The current estimate is speculative and based on the papers read. In addition, we need a separate data set for validation, preferably multiples as we iterate. Each set of validation data should be at least 25% of the size of the training dataset and we should have at least two validation sets. So we need at minimum 15,000 images and preferably 150,000 images. Finally, we expect to need several iterations of most parts of the code in order to achieve the results. We can start with 100-1000 images for initial development, but we should up the number of images as we reach the machine learning stage.

This project will be written in C++ and Python depending on the feature responsibilities.

## Neural Network - Qihan

[Convolutional Neural Network][Convolutional Neural Network]
[Convolutional Neural Network]: https://en.wikipedia.org/wiki/Convolutional_neural_network
The neural network to be used is still under investigation with candidates like Tensorflow being currently ruled out.
This component will be written in C++

## Scoring Function

The scoring function is used to estimate how accurate a piece of line is to another. The scoring function may also be referred to as an evaluation function. To make a good scoring function, the results should be consistent and higher (or lower as long as it’s consistent) scores would imply a better result. One naive way to do this is to just sum up the absolute difference of every pixel. This means the exact image will result in a score of zero, but a completely different piece of lineart will have a fairly high score since there are a lot of different pixels. However, this is a trap. Considering the following example. Each '_' refers to a white pixel and each 'X' refers to a black pixel.

_X__

_X__

_X__

_X__

This 4x4 example image would be a single black line down the second column. Now if we use the naive evaluation function against an image where the black line is down the third column like the following:

__X_

__X_

__X_

__X_ 

We would get a score of 256x8. Each black pixel that should be white would add 256, each white pixel that should be black will add 256, resulting in a score of 2048. Now compare this score to a completely white image. The white image would have a score of 256x4 or 1024 since there’s only white pixels that should be black. As a result the white image would have a lower score than a black line down the third column, thus the function would say the white image is better. However, to a human, the line down the third column would look more similar to the original. This would create an extremely strong local minima and the neural network would usually learn to just spit out an white image. Thus, a good scoring function is really important to making this work. One requirement of the scoring function is that the sketch must score better than a completely white image and higher than a different image. In fact, every image that scores better than the initial sketch should be a better result than the sketch (but not necessarily vice versa). The exact original image should score the best.

### Scoring Using An Image Pyramid - Ian
For each image, we create another one scaled to half the height and width. Then we create another scaled down image of the scaled image and repeat until we hit a minimum width and height. Then we do the naive difference method noted above. However, each pixel in a scaled down image should be weighted higher than an unscaled image, at least 4 times higher for each layer. In addition, the difference should be squared since we want to punish higher differences more than small ones. Thus the weight of each pixel should be:

`sigma_of_l(sigma_of_i((i-i')^2))*s^l`

Where i is the pixel value of the original image, i'is the value of the testing image, s is the scaling factor, and l is the layer of the pyramid. In English, for every pixel of every layer of the pyramid, we take the difference of between the pixels. Square this difference because we want to punish higher differences more. Then, we scale the this value depending on the layer because each layer is smaller than the previous.

If the scaling factor is 4, then each layer of the pyramid would have the same weight. I expect increasing s will cause the neural network to learn rougher approximations that looks good if scaled down. This might be a good thing if I just learn to draw in very high pixel counts. Lowering the scaling factor would cause fully sized image to be valued more, potentially causing the algorithm to give up and spit out completely white images.

There is a slight issue of a simple image pyramid. If a line is one pixel thick, the downscaling might not be as effective. Using the two toy images from above, if we just scale down the two images, we get:

O_		_O

O_	and 	_O

Where O is a gray pixel (the average of two completely black pixels and two white pixels). There is no overlap in this case. The squaring of the differences during the weight calculations would help, but I don’t think this is enough. We should probably use a [Gaussian Pyramid][gaussian] instead to account for this. 

[gaussian]: https://en.wikipedia.org/wiki/Gaussian_blur

### Histogram Analysis - Ian

Using a histogram of pixel intensities might be a good indicator of similarity. This won’t be good enough on its own but could be used as a supplemental signal.

## Sketch Generator

The goal of the sketch generator is to generate something close enough to a sketch so that the neural network can learn the pattern between sketches and cleaned line art. It doesn’t have to be perfect as long as it gets the core differences across. The sketch generator can be run multiple times with different seeds (assuming the algorithm is based off of something random) to generate multiple sketches from each image to increase the size of the dataset.

### Noise - Ian

By adding noise to the original image, we can start the training without actual sketches. After all, cleaning a sketch is pretty much noise removal. Some methods can include Gaussian blur, Gaussian noise, and Jpeg encoding. This is good for initial training and bootstrapping for more data. However, we expect that we will need better data to mimic rough sketches and train the neural network. 


### Translational Method - Ian

This method is just translating the image a small amount then averaging the image it produces against other translated images. Pretty simple method and emulates a bunch of lines roughly centered on the correct lines. However, this method might be too simplistic and might cause the neural network to learn things that aren’t exactly the results we want. 

This method can be improved by patching different areas with different seeds. For example, we can select a random sub rectangle of the image and have that rectangle consists of random small horizontal translations. Then we can select a different rectangle and use vertical translations. This can be extrapolated to a set of averages of a random gaussian distribution of translations at random sub-sections of the image. If we do this, it might be good enough as an initial starting point for data generation.

### Stroke Emulator 1 - Qihan

This method is to emulate various strokes. The first step is to generate a set of images as a sample of the strokes (around 30?) used during a sketch. The algorithm will then take the set of pregenerated strokes as well as as the original artwork. Then, iteratively, it will randomly select a stroke and try to match that stroke in a way that maximizes the score from the scoring function. The stroke can be deformed in the following methods:

* X-translation

* Y-Translation

* Rotation

* Intensity change - From 1 to 1, though we might want to avoid the values too close 0. A negative value would emulate an eraser.

* Scaling - From 0 to 1, we only want to scale the stroke down. Again, maybe we want to avoid the values too close to 0.

This leaves us with a five dimensional space for stroke deformations. Since we only want a rough sketch, we don’t have to actually find the optimal way to draw the image. Thus we can use a gradient descent down this 5 dimensional space to find a local optimum. When placing the stroke down, we can add a bit of random noise (gaussian noise preferably) across all 5 dimensions to emulate an imperfect human drawing the line. Then, we can repeatedly draw strokes until we get a result that hits a certain similarity through the evaluation function. Note that the randomness we add when placing the stroke down might mean the program won’t terminate, so that must be accounted for. We can control how accurate each generated sketch would be to the original by controlling the number of strokes the computer can use and the minimum target accuracy before it stops.

There is the possibility of simulating pressure sensitivity or using vectored strokes to improve this emulation. Another idea is to use machine learning (SVM perhaps?) to learn where to place each stroke.

### Stroke Emulator 2

Another way of emulating strokes is to have the computer try to draw strokes instead of using pre generated images. In this case, we use give a variable momentum and pressure to each stroke to simulate a hand holding a pen. Though, I believe this is too complex for what we want to do.

One possibility is using edge detection to get a rough outline and rebrushing over the edge detected image with thicker and thinner strokes to better approximate a sketch.

## Preprocessor - Ian

Cleans up the images and resizes them if necessary. The preprocessor will read in images from a folder. Then it will convert the image to grayscale (if necessary) and clean the image by making sure the white areas are white and the lines are black. This can be done by looking at a histogram of the intensities. If the picture is dirty, then there will be high counts of gray values in the histogram. We want to select a white threshold (every pixel above a certain intensity should be set to a value of 256) and black threshold (every pixel below the value should be set to 0). After applying the threshold, we can scale the remaining pixels to smooth out the grays in the picture. Finally, we rename the image and save it in another folder as a .png file.

The preprocessor should also generate the gaussian image pyramid we use for the evaluation function.

## Bootstrapper - Ian

The goal of the bootstrapper is to generate more images from a smaller data set. This step isn’t necessary, but can potentially increase our sample size significantly and relieve a lot of burden in finding and downloading suitable sample images. 

Possible ways of doing this includes:

* Flipping the images horizontally

* Flipping the images vertically^1

* Rotating the image^1

* Scaling the image^2

* Translating the image^2

Note 1: Flipping the image vertically or rotating the image might create different semantic data. For example, a neural network trained on upside down images might not be able to internalize eyes should be above the mouth because it is trained on images that have eyes below the mouth.

Note 2: The final dataset should be scale and translation invariant. However, it is good to test some scaled and translated images to make sure it actually doesn’t impact quality. If there is a significant difference, we can feed in some scaled and translated images to help improve that.

The sketch cleaner takes a complete piece of lineart as input, it should spit out the same image as output. Therefore, we can use the original artwork as extra input for training.

## Other Considerations

### Data Gathering

A good neural network can easily need up to millions of input data. Downloading a million images is tedious and not exactly viable unless we have a crawler. However, we think one hundred thousand images should be enough. We might be able to get away with as few as ten thousand. The bootstrapper and multiple runs of the sketch generator can artificially inflate our sample size. Assuming we flip the image horizontally (2x), flip it vertically (2x), do two rotations of the image (3x), and generate 3 sketches from each image with different parameters (3x), this will give us 36 samples per image. This means the minimum number of images we need is 300, preferably 3000. However, note that the bootstrapped images might not be as good as an actual separate training sample.

### Image Types and Quality

While any lineart artwork would work, those with shading may add a confounding factor we aren't able to deal with yet.

### Generating Lineart from Colored Images

### Duplicate Images

### Image Reader and Library

* OpenCV?

* Python Imaging Library?

### Tensorflow

### Google Cloud Stuff

Allows us to run at 5 standard machines non-stop for the first two months. 
