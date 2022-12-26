import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    def fun1(x_):
        return x_[2]
    bboxes.sort(key=fun1)
    row1 = []
    row2 = []
    bound = bboxes[0][2]
    for i in bboxes:
        if i[0] > bound:
            def fun2(x_):
                return x_[1]
            row2.sort(key=fun2)
            row1.append(row2)
            bound = i[2]
            row2 = []
        row2.append(i)

    def fun3(x_):
        return x_[1]
    row2.sort(key=fun3)
    row1.append(row2)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    cropped_box = []
    cropped_box_list = []
    for r in range(len(row1)):
        cropped_box_list = []
        for i in row1[r]:
            character = bw[i[0]:i[2], i[1]:i[3]]
            character = np.pad(character, (30, 30), 'constant', constant_values=(1, 1))
            character = skimage.transform.resize(character, (32, 32))
            character = np.transpose(skimage.morphology.erosion(character))
            cropped_box_list.append([character.flatten()])
        cropped_box.append(cropped_box_list)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    ##########################
    ##### your code here #####
    ##########################
    print("\n" + img + "\n")
    op_str = ""
    for j in range(len(cropped_box)):
        for i in cropped_box[j]:
            h1 = forward(i, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            op_str += letters[np.argmax(probs)][0]
        print(op_str)
        op_str = ""

