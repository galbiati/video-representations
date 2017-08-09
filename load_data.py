import os
import tensorflow as tf

def read_record(filepath_queue):
    """
    Parses a record from tfrecords file

    Args:
    --------
    :filepath_queue comes from tf.train.string_input_producer()

    Outputs:
    -------
    :video is a tensor representation of a single video, sliced to 128 frames
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filepath_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'video': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        }
    )

    video = tf.decode_raw(features['video'], tf.uint8)

    video_shape = tf.stack([-1, 60, 80, 3])
    video = tf.cast(tf.reshape(video, video_shape), tf.float32)
    video = tf.slice(video, [0, 0, 0, 0], [128, -1, -1, -1])

    return video

def inputs(split_type, batchsize, num_epochs, seqlen=64, queue_name=None):
    """
    Queues inputs and targets in batches

    // Future: make shuffling optional with default shuffle=True kwarg

    Args:
    -------
    :split_type determines which split files to refer to
                should be one of 'training', 'testing', or 'validation'
    :batchsize is batchsize
    :num_epochs is the number of times the queue can loop through
    :queue_name is used to create multiple queues for feeding different graph
                sections separately
    Outputs:
    -------
    :video_inputs are batches of videos (rank 5 tensors; batchsize, sequence length, x, y, channels)
    :video_outputs are the same, but offset by one frame
    """
    if not queue_name:
        queue_name = split_type

    data_dir = os.path.expanduser('~/Insight/video-representations/frames')

    if not num_epochs:
        num_epochs = None

    filepath = os.path.join(data_dir, '{}.tfrecords'.format(split_type))

    with tf.name_scope('input/' + queue_name):
        filepath_queue = tf.train.string_input_producer([filepath], num_epochs=num_epochs)

    video = read_record(filepath_queue)
    videos = tf.train.shuffle_batch(
        [video], batchsize,
        capacity=128 + 2*batchsize, min_after_dequeue=128, num_threads=2
    )

    video_inputs = tf.slice(videos, begin=[0, 0, 0, 0, 0], size=[-1, seqlen, -1, -1, -1])
    video_outputs = tf.slice(videos, begin=[0, 1, 0, 0, 0], size=[-1, seqlen, -1, -1, -1])

    return video_inputs, video_outputs
