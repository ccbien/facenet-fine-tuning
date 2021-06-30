import sys
sys.path.append('model')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from inception_resnet_v1 import InceptionResNetV1

def get_l2_norm_model(model):
    inp = model.input
    out = tf.math.l2_normalize(model.output, axis=1)
    return Model(inp, out)