import pickle

from PIL import Image
from glob import glob

import random
import numpy

from torch import Tensor

import os
import sys

currdir = os.path.dirname(sys.argv[0])
os.chdir(os.path.abspath(currdir))


def get_data(hm_samples):
    files = glob('data/*.jpg')
    data = []
    for file in random.choices(files, k=hm_samples):
        img = Image.open(file) ; img.load()
        data.append(Tensor(numpy.asarray(img).reshape(3, 256, 256)))
    return data

def batchify(resource, batch_size):
    hm_batches = int(len(resource) / batch_size)
    batched_resource = [resource[_ * batch_size : (_+1) * batch_size]
                        for _ in range(hm_batches)]
    hm_leftover = len(resource) % batch_size
    if hm_leftover != 0:
        batched_resource.append(resource[-hm_leftover:])

    return batched_resource

def pickle_save(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(MacOSFile(f))
    except: return None

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size
