import os
import numpy as np
import tensorflow as tf
import tqdm
from training import *
from render import *
from download import get_filepaths, video_to_array
from models.AELSTM import *
from models.model import *
L = tf.layers

def infer(video_files, model_file, seqlen=64, batchsize=8):
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

    if len(video_files) > 8:
        # for now, don't handle more than 8 videos at once
        print('Please use 8 or fewer videos at a time')
        return None

    # load videos to arrays
    video_arrays = [video_to_array(vf, n_frames=seqlen+1) for vf in video_files]
    video_arrays = np.concatenate(video_arrays)

    # initialize downsampler and model

    with tf.device('/cpu:0'):
        video_placeholder = tf.placeholder(name='video', dtype=tf.float32, shape=(None, 240, 320, 3))
        downsampled = tf.tf.image.resize_images(video_placeholder, [60, 80])
        downsampled = tf.reshape(downsampled, (batchsize, seqlen+1, 60, 80, 3))


    model = Model(encoder, lstm_cell, tied_decoder, batchsize, seqlen)

    input_videos = tf.slice(downsampled, begin=[0, 0, 0, 0, 0], size=[batchsize, seqlen, -1, -1, -1])
    output_videos = tf.slice(downsampled, begin=[0, 1, 0, 0, 0], size=[batchsize, seqlen, -1, -1, -1])

    encoded, transitioned, decoded = model.build(input_videos)
    loss = tf.reduce_mean(tf.pow(decoded - output_videos, 2))

    saver = tf.train.Saver()

    with tf.Session() as sesh:
        saver.restore(sesh, model_file)
        loss_value, encodings, transitions, predictions, recovered = sesh.run(
            [loss, encoded, transitioned, decoded, downsampled],
            {video_placeholder: video_arrays}
        )


    return loss_value, encodings, transitions, predictions, recovered

def main():
    data_dir = os.path.expanduser('~/Insight/video-representations/')
    outputs_dir = os.path.join(data_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    model_file = 'ptypelstm-tied-relu'
    video_files, _ = get_filepaths(data_dir)
    np.random.shuffle(video_files)
    video_files = video_files[:8]

    loss, encodings, transitions, predictions, recovered = infer(video_files, model_file)
    outputs_filename = '{}_output.mp4'
    recover_filename = '{}_recov.mp4'

    for i in range(8):
        render_movie(predictions[i], os.path.join(outputs_dir, outputs_filename.format(i)))
        render_movie(recovered[i], os.path.join(outputs_dir, recover_filename.format(i)))

    np.savez_compressed(os.path.join(outputs_dir, 'infered.npz'), loss, encodings, transitions)


    return None


if __name__ == '__main__':
    main()
