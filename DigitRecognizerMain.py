#
# Verify Reading Dataset via MnistDataloader class
# Modified by Harlan Howe from the original found at
# https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=train-labels-idx1-ubyte
#
# %matplotlib inline
import random
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
from os.path import join

import numpy as np

from MnistDataloaderFile import MnistDataloader

#
# Set file paths based on added MNIST Datasets
#
input_path = 'input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

images_train = []
labels_train = []
images_test = []
labels_test = []


def show_images(images: List[np.ndarray], title_texts: List[str]) -> None:
    """
    Helper function to show a list of images with their relating titles
    part of the starter code that came from
    https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=train-labels-idx1-ubyte
    lightly modified to fit the screen better.
    :param images:  a list of images to display
    :param title_texts: a list of captions matching the images
    :return: None
    """
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(10, 9))
    index = 1
    for x in zip(images, title_texts):
        img = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(img, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=10)
        index += 1
    plt.waitforbuttonpress()


def print_digit(digit_image: np.ndarray) -> None:
    """
    prints a formatted 28 x 28 grid of the numbers stored in digit_image
    :param digit_image: the image we wish to display in the console.
    :return: None
    """
    print("-" * (4*28-1))
    for row in digit_image:
        for item in row:
            print(f"{item:3} ", end="")
        print("")
    print("-" * (4*28-1))


#
# Load MINST dataset
#
def load_data():
    """
    load the data from the files into global images_train, labels_train, images_test, and labels_test.
    from the starter code at https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=train-labels-idx1-ubyte
    but with different list names.
    :return: None
    """
    global images_train, labels_train, images_test, labels_test
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (images_train, labels_train), (images_test, labels_test) = mnist_dataloader.load_data()
    print("Data loaded from file.")


#
# Show some random training and test images
#
def display_random_sample() -> None:
    """
    picks a random image from training data and prints it, then picks 10 random training images and 5 random test images
    and displays them graphically in a new window, until the user presses a key.
    :return: None
    """
    idx = random.randint(1, len(images_train))
    print_digit(images_train[idx])
    print(f"This is a {labels_train[idx]}.")

    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(images_train[r])
        titles_2_show.append('train[' + str(r) + '] = ' + str(labels_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(images_test[r])
        titles_2_show.append('test[' + str(r) + '] = ' + str(labels_test[r]))

    show_images(images_2_show, titles_2_show)


def test_identifier() -> None:
    """
    Asks user for a number of digits to test, then picks that many test_images, predicts their value
    and displays the digits and the predicted number in a new window until the user presses a key.
    :return: None
    """
    try:
        N = int(input("How many digits to test? "))
    except ValueError:
        N = 1

    try:
        pool_size = int(input("How large a training sample should I use? "))
    except ValueError:
        pool_size = 500

    digit_list: List[np.ndarray] = []
    label_list: List[str] = []
    for i in range(N):
        idx = random.randint(0, len(images_test))
        guess = identify_digit(images_test[idx], pool_size)
        digit_list.append(images_test[idx])
        label_list.append(f"I think this is {guess}.")
    print("Click in the graphics window and press any key to continue.")
    show_images(digit_list, label_list)


def identify_digit(digit_image: np.ndarray, pool_size: int) -> int:
    """
    attempts to identify which digit 0-9 the given digit_image represents, using pool_size digits randomly
    selected from the training data.
    :param digit_image: a 28 x 28 grid of integers 0-255
    :param pool_size: the number of images to sample from the training data
    :return: a number 0-9, representing the best guess for what the given image represents.
    """
    # TODO: you write this method!

    return 0  # replace this line.


def identify_large_sample() -> None:
    """
    asks the user for a number of identifications to attempt and the size of the pool used to make those identifications
    print the success rate and the time it took.
    :return: None
    """
    try:
        N = int(input("How many digits should I try to identify? "))
    except ValueError:
        N = 2500
    try:
        pool_size = int(input("How many training images should I sample to make the identification? "))
    except ValueError:
        pool_size = 1000

    num_correct = 0
    print("Starting.")

    start_time = datetime.now()
    for i in range(N):
        # TODO: you write this... pick a random test image, identify what digit you think it will be and compare to the
        #       actual label for this test image. If they match, increment num_correct.
        pass

    end_time = datetime.now()
    print(f"Score: {num_correct}/{N} = {(100*num_correct/N):3.2f}% using pool = {pool_size} in {end_time-start_time}"
          f" (Hours : Minutes : Seconds)")


if __name__ == "__main__":
    load_data()
    display_random_sample()
    test_identifier()
    identify_large_sample()
