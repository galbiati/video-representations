import tensorflow as tf
import numpy as np

def get_l2_term(exclude_names=['lstm']):
    l2_vars = [v for v in tf.trainable_variables() if 'bias' not in v.name]
    for name in exclude_names:
        l2_vars = [v for v in l2_vars if name not in v.name]
    return tf.add_n([tf.nn.l2_loss(v) for v in l2_vars])


def train(model, inputs, targets,
            num_epochs=10, batchsize=16, l2weight=None,
            save=True, print_interval=500):

    encoded, transitioned, decoded = model.build_full(inputs, batchsize)
    loss = tf.reduce_mean(tf.pow(decoded - targets, 2))
    training_loss = loss + l2weight * get_l2_term() if l2weight is not None else loss
    train_step = tf.train.AdamOptimizer().minimize(training_loss)

    if save:
        saver = tf.train.Saver()

    init_global = tf.global_variables_initalizer()
    init_local = tf.local_variables_initializer()
    coord = tf.train.Coordinator()

    with tf.Session() as sesh:
        sesh.run([init_global, init_local])
        threads = tf.train.start_queue_runners(sess=sesh, coord=coord)

        losses = []

        try:
            step = 0

            while not coord.should_stop():
                _, loss_value = sesh.run([train_step, loss])
                losses.append(loss_value)

                if step % 500 == 0:
                    print('Step {} loss:\t{:.8f}'.format(step, loss_value))

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done; loss:\t{:.8f}'.format(loss_value))

        finally:
            coord.request_stop()

        coord.join(threads)
        saver.save(sesh, 'prototype-lstm')

    return losses

def iterate_minibatches(inputs, targets, batchsize=32, shuffle=False):
    """Utility func for shuffling minibatches with placeholder inputs"""
    assert len(inputs) == len(targets)
    if shuffle:
        idxs = np.arange(len(inputs))
        np.random.shuffle(idxs)

    for idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = idxs[idx:idx + batchsize]
        else:
            excerpt = slice(idx, idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
