import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    ##########################
    ##### your code here #####
    ##########################
    est_sig = skimage.restoration.estimate_sigma(image, multichannel=False)
    denoise = skimage.restoration.denoise_wavelet(image, 4*est_sig, multichannel=True)
    greyscale = skimage.color.rgb2gray(denoise)
    threshold = skimage.filters.threshold_otsu(greyscale)
    binary = greyscale < threshold
    labels = skimage.measure.label(skimage.segmentation.clear_border(
        skimage.morphology.closing(binary, skimage.morphology.square(7))))
    sample = skimage.measure.regionprops(labels)
    for i in sample:
        if i.area >= 150:
            bboxes.append(i.bbox)
    bw = 1.0 - binary

    return bboxes, bw

