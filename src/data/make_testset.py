import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("./")

from src.utilities import *


def make_dedicated_testset(train_dir, test_dir, test_size):

    amount_of_test_figures = test_size * len(os.listdir(train_dir))
    test_fnames = np.random.choice(os.listdir(train_dir), 10)

    # Now remove the sampled filenames from the train_dir
    # ...

if __name__ == '__main__':
    make_dedicated_testset(train_dir, test_dir, test_size=0.1)