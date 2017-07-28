import os
import numpy as np
import tensorflow as tf
import tqdm
from training import *
from load_data import *
import models.ConvAE as cae

L = tf.layers

# build graph, infer
def infer(X, y, model_file, batchsize=128):
    """X, y are numpy arrays. model_file is from tf.Saver.save"""
    print("Building graph...")
    input_var = tf.placeholder(dtype=tf.float32, shape=(None, 3, 60, 80), name='input')
    target_var = tf.placeholder(dtype=tf.float32, shape=(None, 3, 60, 80), name='target')
    l2_weight = .01

    with tf.variable_scope('encoder'):
        encoded = cae.encoder(input_var)

    with tf.variable_scope('decoder'):
        decoded = cae.decoder(encoded)

    l2_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    loss = tf.reduce_mean(tf.pow(decoded - target_var, 2))
    train_step = tf.train.AdamOptimizer().minimize(loss + l2_weight*l2_term)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sesh:
        print('Restoring params...')
        saver.restore(sesh, model_file)

        outputs = []
        prog_bar = tqdm.tqdm(
            iterate_minibatches(X, y, batchsize=batchsize, shuffle=False),
            total=X.shape[0]//batchsize, unit='batches', desc='Completed minibatches'

        )
        for batch in prog_bar:
            X_batch, y_batch = batch
            output = sesh.run(decoded, {input_var: X_batch})
            outputs.append(output)

    return np.concatenate(outputs)

def main():
    data_dir = os.path.expanduser('~/Insight/video-representations/data/downsampled')
    model_file = 'tmp/models/prototype_ae initial.ckpt'
    X, y = load_all_data_stacked(data_dir, skip_first=1, every_n=2)

    predictions = infer(X, y, model_file)
    print('Saving predictions...')
    np.savez_compressed(os.path.join(data_dir, 'predictions.npz'), predictions)


if __name__ == '__main__':
    main()
