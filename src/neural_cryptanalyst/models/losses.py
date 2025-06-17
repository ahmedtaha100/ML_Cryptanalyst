import tensorflow as tf
from tensorflow.keras import backend as K

def ranking_loss(y_true, y_pred):
    y_true_idx = tf.argmax(y_true, axis=-1)
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([tf.range(batch_size), tf.cast(y_true_idx, tf.int32)], axis=1)
    true_scores = tf.gather_nd(y_pred, indices)
    loss = -tf.math.log(true_scores / tf.reduce_sum(y_pred, axis=-1))
    return tf.reduce_mean(loss)

def focal_loss_ratio(alpha=0.25, gamma=2.0):
    def flr(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true_idx = tf.argmax(y_true, axis=-1)
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(batch_size), tf.cast(y_true_idx, tf.int32)], axis=1)
        p_t = tf.gather_nd(y_pred, indices)
        focal_pos = -alpha * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
        focal_neg = 0
        for i in range(256):
            mask = tf.cast(tf.not_equal(y_true_idx, i), tf.float32)
            p_neg = y_pred[:, i]
            focal_neg += alpha * tf.pow(p_neg, gamma) * tf.math.log(1 - p_neg) * mask
        return focal_pos / (focal_neg + epsilon)
    return flr

def cross_entropy_ratio(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true_idx = tf.argmax(y_true, axis=-1)
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([tf.range(batch_size), tf.cast(y_true_idx, tf.int32)], axis=1)
    p_true = tf.gather_nd(y_pred, indices)
    ce_true = -tf.math.log(p_true)
    ce_false = 0
    for i in range(256):
        mask = tf.cast(tf.not_equal(y_true_idx, i), tf.float32)
        ce_false += -tf.math.log(1 - y_pred[:, i]) * mask
    return ce_true / (ce_false + epsilon)
