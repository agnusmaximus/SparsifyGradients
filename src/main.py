from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import train

import time
import subprocess
import numpy as np
import os
import sys
import tensorflow as tf
import cifar10
import cifar10_input
from mpi4py import MPI

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', .1,
                          """Constant learning rate""")
tf.app.flags.DEFINE_integer('num_epochs', 10,
                            """Number of passes of data""")

def main(argv=None):
    cifar10.maybe_download_and_extract()
    train.train()

if __name__ == '__main__':
    tf.app.run()
