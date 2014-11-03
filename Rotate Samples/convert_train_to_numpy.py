import numpy as np
import csv
from scipy import ndimage
from scipy import misc

# Read the png images and converts them to a numpy array.
# Also rotates them in 6 orientations and saves the copies.
train_inputs = []
for i in range (1,50001,1):
    file_name = "" + str(i) + ".png"
    img_array = ndimage.imread(file_name)
    train_inputs.append(img_array.flatten())
    for i in range (60,360,60):
        rotated_array = ndimage.rotate(img_array, i, axes=(1, 0), reshape=False)
        train_inputs.append(rotated_array.flatten())

train_inputs_np = np.asarray(train_inputs)
print np.shape(train_inputs_np)
np.save('train_inputs', train_inputs_np)

