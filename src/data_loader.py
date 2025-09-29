import numpy as np
from idx2numpy import convert_from_file

def load_full_mnist_dataset():
    """
    Loads the full MNIST dataset from the local IDX files.

    Returns:
        A tuple containing (all_images, all_labels).
    """
    try:
        train_images_raw = convert_from_file('train-images-idx3-ubyte')
        train_labels_raw = convert_from_file('train-labels-idx1-ubyte')
        test_images_raw = convert_from_file('t10k-images-idx3-ubyte')
        test_labels_raw = convert_from_file('t10k-labels-idx1-ubyte')
    except FileNotFoundError:
        print("ERROR: MNIST data files not found.")
        print("Please download them from http://yann.lecun.com/exdb/mnist/ and place them in the root project directory.")
        return None, None

    all_images = np.concatenate([train_images_raw, test_images_raw])
    all_labels = np.concatenate([train_labels_raw, test_labels_raw])

    # Flatten and normalize
    all_images_flattened = all_images.reshape(-1, 28*28)
    all_images_normalized = all_images_flattened.astype(np.float32) / 255.0

    return all_images_normalized, all_labels