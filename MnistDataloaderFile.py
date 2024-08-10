#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
# Modified by Harlan Howe from the original found at
# https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=train-labels-idx1-ubyte
#
import numpy as np  # linear algebra
import struct
from array import array


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))  # read 8 bytes and split them into 2 4-byte numbers.
            # the "magic" number comes from a code, 8, meaning "unsigned byte" to explain the type of data in the file;
            # this is multiplied by 256, and we add the number of dimensions in this dataset: 8 * 256 + 1 = 2049.
            # the next byte, "size," is the number of items in the one direction we have.
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {format(magic)}.")
            labels = array("B", file.read())  # read the rest of the file, treating it as unsigned bytes ("B") and load
            #                                   it into the array, labels.

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))  # by the same reasoning as before,
            #  magic is 8 * 256 + 3 = 2051, so "8" means unsigned bytes, and the data is 3-dimensional.
            #  size is the number of items in the first dimension (in this case, the number of digit images);
            #  rows is the number of items in the second dimension (in this case, the number of rows per image);
            #  cols is the number of items in the third dimension (in this case, the number of columns per image).
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {format(magic)}.")
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            # grabs 784 of the unsigned ints from image_data and converts them to ints.
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=int)
            # converts 1-d list of 784 ints to 2-d (28 x 28) list of lists of ints.
            img = img.reshape(28, 28)
            # puts this image into the list of images.
            images.append(img)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)