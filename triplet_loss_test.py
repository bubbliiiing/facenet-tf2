import tensorflow as tf
from tensorflow.keras import backend as K


def triplet_loss(y_true, y_pred, alpha=1):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.sum(K.maximum(basic_loss, 0))
    return loss

def triplet_loss_test():
    y_true = (None, None, None)
    y_pred = (tf.random.normal([3, 128], mean=1, stddev=0.1, seed=1),
                tf.random.normal([3, 128], mean=1, stddev=0.1, seed=1),
                tf.random.normal([3, 128], mean=0, stddev=0.1, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.numpy()))

def main():
    triplet_loss_test()

if __name__ == '__main__':
    main()
