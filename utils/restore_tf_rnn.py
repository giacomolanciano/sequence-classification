import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils.constants import TRAINED_MODELS_FOLDER


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.join(TRAINED_MODELS_FOLDER, '1494508907.1354291.ckpt.meta'))
    saver.restore(sess, os.path.join(TRAINED_MODELS_FOLDER, '1494508907.1354291.ckpt'))
