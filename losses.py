import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import shape

def CustomTripletLoss(margin=0.2):
    def loss_function(true, y):
        yP = true[:, :128]
        yN = true[:, 128:]
        pos = K.sqrt(K.sum((y - yP)**2, axis=1))
        neg = K.sqrt(K.sum((y - yN)**2, axis=1))
        loss = tf.reduce_mean(pos - neg) + margin
        return K.maximum(loss, 0)
    return loss_function