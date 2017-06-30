import os
import numpy as np
import struct
import gzip
import itertools

from .dataset import DataSet


class MnistDataSet(DataSet):

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = "/tmp/"

        self.train_fname_images = data_dir + 'train-images-idx3-ubyte.gz'
        self.train_fname_labels = data_dir + 'train-labels-idx1-ubyte.gz'
        self.test_fname_images = data_dir + 't10k-images-idx3-ubyte.gz'
        self.test_fname_labels = data_dir + 't10k-labels-idx1-ubyte.gz'

        if not os.path.exists(self.test_fname_labels):
            raise FileExistsError(
                "file not found, try\nwget "
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz "
                "-O {} ".format(data_dir))

        if not os.path.exists(self.test_fname_images):
            raise FileExistsError(
                "file not found, try\nwget "
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz "
                "-O {} ".format(data_dir))

        if not os.path.exists(self.train_fname_labels):
            raise FileExistsError(
                "file not found, try\nwget "
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz "
                "-O {} ".format(data_dir))

        if not os.path.exists(self.train_fname_images):
            raise FileExistsError(
                "file not found, try\nwget "
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz "
                "-O {} ".format(data_dir))

    def _read_labels(self, fname):
        f = gzip.GzipFile(fname, 'rb')
        magic_nr, n_examples = struct.unpack(">II", f.read(8))
        labels = np.fromstring(f.read(), dtype='uint8').reshape(n_examples, 1)
        return labels

    def _read_images(self, fname):
        f = gzip.GzipFile(fname, 'rb')
        magic_nr, n_examples, rows, cols = struct.unpack(">IIII", f.read(16))
        shape = (n_examples, rows*cols)
        images = np.fromstring(f.read(), dtype='uint8').reshape(shape)
        return images

    def itrain_data(self, start, stop, step):
        images = self._read_images(self.train_fname_images)/255
        labels = self._read_labels(self.train_fname_labels)

        return itertools.islice(zip(images, labels), start, stop, step)

    def itest_data(self, start, stop, step):
        images = self._read_images(self.test_fname_images)/255
        labels = self._read_labels(self.test_fname_labels)

        return itertools.islice(zip(images, labels), start, stop, step)
