import os
import numpy as np
import moviepy.editor as mpe

import tensorflow as tf
from tensorflow.python.framework import ops     # is this needed?
import load_data as load
from models.model import Model
from models.customlayers import *               # is this needed?
from models.activations import *                # is this needed?

from models.AELSTM import *


def infer(split_type, batchsize=4, seqlen=64):
    """
    Loads a pretrained model and performs inference on the provided dataset
    """
    try:
        assert split_type != 'training'
    except:
        raise ValueError('records file too big for memory!')

    # create the model
    model = Model(encoder, lstm_cell, decoder, batchsize, seqlen)
    inputs, targets = load.inputs(split_type, batchsize, seqlen)
    encoded, transitioned, decoded = model.build(inputs)
    loss = tf.reduce_mean(tf.pow(decoded - inputs, 2))

    # prepare graph
    saver = tf.train.Saver()
    init_global = tf.global_variables_initalizer()
    init_local = tf.local_variables_initializer()
    coord = tf.train.Coordinator()

    # run inference
    with tf.Session() as sesh:
        sesh.run([init_global, init_local])     # initialize graph
        saver.restore(sesh, 'ptypelstm')        # restore saved params
        threads = tf.train.start_queue_runners(sess=sesh, coord=coord)

        recoveries = []
        predictions = []
        encodings = []
        transitions = []
        losses = []

        try:
            # can this be converted to for loop? would be nice to tqdm
            step = 0
            while not coord.should_stop():
                recovery, prediction, encoding, transition, loss_value = sesh.run(
                    [targets, decoded, encoded, transitioned, loss]
                )
                losses.append(loss_value)
                recoveries.append(recovery)
                predictions.append(prediction)
                encodings.append(encoding)
                transitions.append(transition)

        except tf.errors.OutOfRangeError:
            print('Inference finished!')

        finally:
            coord.request_stop()

        coord.join(threads)
        print('Average loss: {:.3f}'.format(np.mean(losses)))

        np.savez_compressed('recovery.npz', *recoveries)
        np.savez_compressed('predictions.npz', *predictions)
        np.savez_compressed('encodings.npz', *encodings)
        np.savez_compressed('transitions.npz', *transitions)

    # return...np.ndarrays?

    return losses, predictions

def render_movie(frame_array, output_file, fps=5, max_pixel=255):
    """
    Render a movie from an array of RGB frames to output_file

    Args:
    -------
    :frame_array is a numpy array of dimensions (frame, x, y, channel)
    :output_file is where to save movie
    :fps is movie frame rate
    :max_pixel is a clipping value
    """
    assert frame_array.ndim == 4
    n_frames = frame_array.shape[0]
    clipped_frames = np.minimum(np.maximum(frame_array, 0), max_pixel)
    clip = mpe.ImageSequenceClip(list(clipped_frames), fps=fps)
    clip.write_videofile(output_file)
    return None

def main():
    """Render videos from the test set"""
    losses, predictions = infer('testing')
    for i, array in enumerate(predictions):
        render_movie(array, 'v_{}.mp4'.format(i))

if __name__ == '__main__':
    main()
