import os
import numpy as np
import tensorflow as tf
import tqdm
from training import *
from download import *
from models.AELSTM import *
L = tf.layers

def infer(video_files, model_file, output_dir, seqlen=64, batchsize=8):
    """
    Takes a list of avi files and a runs the model in model_file on them

    Args:
    ------
    :video_files is a list of filepaths to avi files
    :model_file is the name of a saved tensorflow session
    :output_dir is where the outputs and logs will be stored
    :batchsize is batchsize

    Outputs:
    -------
    None
    """

    # load videos to arrays
    video_arrays = [video_to_array(vf, n_frames=seqlen+1) for vf in video_files]
    video_arrays = np.concatenate(video_arrays)

    # initialize downsampler and model

    with tf.device('/cpu:0'):
        video_placeholder = tf.placeholder(name='video', dtype=tf.float32, shape=(None, 240, 320, 3))
        downsampled = tf.layers.max_pooling2d(video_placeholder, 4, 4, name='downsampler')
        downsampled = tf.reshape(downsampled, (-1, seqlen+1, 60, 80, 3))


    model = Model(encoder, lstm_cell, tied_decoder, batchsize, seqlen)

    inputs = tf.slice(downsampled, begin=[0, 0, 0, 0, 0], size=[-1, seqlen, -1, -1, -1])
    outputs = tf.slice(downsampled, begin=[0, 1, 0, 0, 0], size=[-1, seqlen, -1, -1, -1])

    encoded, transitioned, decoded = model.build(inputs)
    loss = tf.reduce_mean(tf.pow(decoded - outputs, 2))

    saver = tf.train.Saver()

    with tf.Session() as sesh:
        saver.restore(sesh, model_file)
        loss_value, encodings, transitions, predictions = sesh.run(
            [loss, encoded, transitioned, decoded],
            {video_placeholder: video_arrays}
        )


    return loss_value, encodings, transitions, predictions

def main():
    data_dir = os.path.expanduser('~/Insight/video-representations/data/downsampled')
    model_file = 'tmp/models/prototype_ae initial.ckpt'

    return None


if __name__ == '__main__':
    main()
