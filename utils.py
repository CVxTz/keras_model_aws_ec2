from random import shuffle
from io import StringIO
from PIL import Image
import numpy as np  # linear algebra
from tensorflow.keras.applications.nasnet import preprocess_input
import os
import base64


def get_class_from_file_path(file_path):
    return 1 if "dog" in file_path.split('/')[-1] else 0


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_image(path):
    img = Image.open(path).resize((64, 64)).convert("RGB")
    img = np.array(img)
    return img


def read_image_base64(b64):
    file_like = StringIO(b64)
    img = Image.open(file_like).resize((64, 64)).convert("RGB")
    img = np.array(img)
    return img


def encode_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return encoded_string#.decode("ascii")


def data_gen(list_files, batch_size):
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [read_image(x) for x in batch]
            Y = [get_class_from_file_path(x) for x in batch]

            X = [preprocess_input(x) for x in X]

            yield np.array(X), np.array(Y)
