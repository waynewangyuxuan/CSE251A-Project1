"""
MNIST Data Loader

Loads MNIST dataset from IDX file format (Kaggle version).
"""

import numpy as np
import struct
import os


def load_mnist(data_dir):
    """
    Load MNIST dataset from IDX files.

    Args:
        data_dir: Directory containing the MNIST files

    Returns:
        X_train: (60000, 784) training images
        y_train: (60000,) training labels
        X_test: (10000, 784) test images
        y_test: (10000,) test labels
    """
    # File names
    train_images_file = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

    # Load training data
    X_train = _load_images(train_images_file)
    y_train = _load_labels(train_labels_file)

    # Load test data
    X_test = _load_images(test_images_file)
    y_test = _load_labels(test_labels_file)

    return X_train, y_train, X_test, y_test


def _load_images(filepath):
    """
    Load images from IDX file format.

    IDX file format:
    - 4 bytes: magic number (2051 for images)
    - 4 bytes: number of images
    - 4 bytes: number of rows
    - 4 bytes: number of columns
    - rest: pixel data (unsigned bytes)
    """
    with open(filepath, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}, expected 2051")

        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)

        # Normalize to [0, 1]
        data = data.astype(np.float32) / 255.0

    return data


def _load_labels(filepath):
    """
    Load labels from IDX file format.

    IDX file format:
    - 4 bytes: magic number (2049 for labels)
    - 4 bytes: number of labels
    - rest: label data (unsigned bytes)
    """
    with open(filepath, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}, expected 2049")

        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


if __name__ == '__main__':
    # Test loading
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    try:
        X_train, y_train, X_test, y_test = load_mnist(data_dir)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Label distribution: {np.bincount(y_train)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download MNIST data from Kaggle and place in data/ directory")
