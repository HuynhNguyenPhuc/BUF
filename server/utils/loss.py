import tensorflow as tf

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mape(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))

def r2_score(y_true, y_pred):
    y_mean = tf.reduce_mean(y_true)
    y_variance = tf.reduce_mean(tf.square(y_true - y_mean))
    return 1 - mse(y_true, y_pred) / y_variance