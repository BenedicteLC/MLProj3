import numpy as np
import csv
from scipy import ndimage
from scipy import misc

# Read the png images and converts them to a numpy array.
# Also rotates them in 6 orientations and saves the copies.
test_inputs = []
for i in range (1,20001,1):
    file_name = "" + str(i) + ".png"
    img_array = ndimage.imread(file_name)    
    test_inputs.append(img_array.flatten())
    for i in range (60,360,60):
        rotated_array = ndimage.rotate(img_array, i, axes=(1, 0), reshape=False)
        test_inputs.append(rotated_array.flatten())

test_inputs_np = np.asarray(test_inputs)
print np.shape(test_inputs_np)
np.save('test_inputs', test_inputs_np)