import os, argparse, gc, subprocess
import numpy as np
import requests as req
import moviepy.editor as mpe
import tensorflow as tf
import patoolib
import tqdm

## Download and extract


def download_videos(download_dir):
    """Download UCF-101 videos"""

    # what goes where
    url = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'
    savefile = os.path.join(download_dir, 'UCF101.rar')

    # streaming request for big file
    with req.Session() as sesh:
        g = sesh.get(url, stream=True)

    # write continuously to file from stream object

    # params
    chunk_size = 1000
    final_size = float(g.headers['content-length']) / chunk_size

    # use tqdm for progress bar
    chunk_iterator = tqdm.tqdm(
        g.iter_content(chunk_size),
        total=final_size, unit='B', unit_scale=True,
        desc='Downloading video archive...'
    )

    # write chunks
    with open(savefile, 'wb') as f:
        for chunk in chunk_iterator:
            f.write(chunk)

    return None

def extract_videos(download_dir, extract_dir):
    """Extract UCF-101 videos from rar archive"""

    filename = os.path.join(download_dir, 'UCF101.rar')
    patoolib.extract_archive(filename, outdir=extract_dir)

    # os.remove(filename)

    return None


## Conver to TFRecords files

def get_filepaths(extract_dir):
    """Get paths of all files in directory"""

    index = []
    labels = []
    _extract_dir = os.path.join(extract_dir, 'UCF-101')
    for folder in os.listdir(_extract_dir):
        labels.append(folder)
        folderpath = os.path.join(_extract_dir, folder)

        if not os.path.isdir(folderpath):
            continue

        for filename in os.listdir(folderpath):
            if 'avi' not in filename:
                continue

            if filename[0] == '.':
                continue

            filepath = os.path.join(folderpath, filename)

            if os.path.exists(filepath):
                index.append(filepath)
            else:
                print(filepath)
    return index, labels

def video_to_array(video_file, n_frames=256):
    """Read video file into a numpy array and reduce framerate"""
    video = mpe.VideoFileClip(video_file)
    video_array = np.array([f for f in video.iter_frames()])
    video.reader.close()
    del video.reader
    del video

    if video_array.shape[0] > n_frames:
        return video_array[:n_frames].astype(np.float32)
    else:
        shape = video_array.shape
        pad = np.zeros([n_frames - shape[0], shape[1], shape[2], shape[3]])
        return np.concatenate([video_array, pad]).astype(np.float32)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def split_video_files(filepaths, ratio=10):
    """
    Splits a list of video files into training, validation, and test sets

    Args:
    - filepaths is a list of complete filepaths for video files in the dataset
    - ratio is the inverted fraction indicating the relative size of validation/test sets

    For example, ratio=10 means 1/10th of the data set aside as test set,
    and 1/10th set aside as validation set
    """

    filepath_array = np.array(filepaths)
    n_files = filepath_array.shape[0]
    filepath_index = np.arange(n_files)
    np.random.shuffle(filepath_index)

    train_filepaths = filepath_array[filepath_index[2*(n_files//ratio):]]
    test_filepaths = filepath_array[:n_files//ratio]
    validation_filepaths = filepath_array[n_files//ratio:2*(n_files//ratio)]

    return train_filepaths.tolist(), validation_filepaths.tolist(), test_filepaths.tolist()

def video_files_to_tfrecords(output_file, filepaths, label_dict):
    """Serializes video files in filepaths to a tfrecords file in output_file"""

    if type(filepaths) != list:
        filepaths = [filepaths]    # catch single inputs (not a normal case)

    tqkws = {
        'total': len(filepaths),
        'unit': ' videos',
        'desc': 'Serializing video frames'
    }

    with tf.device('/cpu:0'):
        video_placeholder = tf.placeholder(name='video', dtype=tf.float32, shape=(None, 240, 320, 3))
        downsampled = tf.layers.max_pooling2d(video_placeholder, 4, 4, name='downsampler')
        downsampled_reshaped = tf.reshape(downsampled, (-1, 60, 80, 3))

    with tf.python_io.TFRecordWriter(output_file) as writer:
        with tf.Session() as sesh:
            for path in tqdm.tqdm(filepaths, **tqkws):
                video_array = video_to_array(path)
                label = label_dict[os.path.split(os.path.abspath(os.path.join(path, os.pardir)))[-1]]

                l = video_array.shape[0]
                w = video_array.shape[2]
                h = video_array.shape[1]

                if h != 240 or w != 320:
                    continue

                downsampled_video_array = downsampled_reshaped.eval({video_placeholder: video_array})
                feature_dict = {
                    'height': _int_feature(h),
                    'width': _int_feature(w),
                    'length': _int_feature(l),
                    'video': _bytes_feature(downsampled_video_array.astype(np.uint8).tostring()),
                    'label': _int_feature(label)
                }

                observation = tf.train.Example(features=tf.train.Features(feature=feature_dict))

                writer.write(observation.SerializeToString())

def main(download_dir, extract_dir, output_dir, downsample_frames=15):

    if not os.path.exists(os.path.join(download_dir, 'UCF101.rar')):
        download_videos(download_dir)

    print('\nExtracting archive...\n')
    # extract_videos(download_dir, extract_dir)

    filepaths, labels = get_filepaths(extract_dir)
    training_filepaths, validation_filepaths, testing_filepaths = split_video_files(filepaths)
    label_dict = dict(zip(labels, np.arange(len(labels))))

    print('\nSerialize me, Scotty')
    os.makedirs(os.path.join(output_dir, 'training'), exist_ok=True)
    for i in range(len(training_filepaths)//100 + 1):
        training_output = os.path.join(output_dir, 'training/training{}.tfrecords'.format(i))
        video_files_to_tfrecords(training_output, training_filepaths[i:100*i], label_dict)

    validation_output = os.path.join(output_dir, 'validation.tfrecords')
    video_files_to_tfrecords(validation_output, validation_filepaths, label_dict)

    testing_output = os.path.join(output_dir, 'testing.tfrecords')
    video_files_to_tfrecords(testing_output, testing_filepaths, label_dict)

    print('\nAll done!')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the UCF-101 video data, extract it, and convert videos to downsampled arrays in npz files')

    parser.add_argument('--download_destination', type=str, help="Where the rar archive goes")
    parser.add_argument('--extract_destination', type=str, help="Where the rar contents are extracted")
    parser.add_argument('--output_destination', type=str, help="Where the tensorflow records are stored")
    args = parser.parse_args()

    # do the below inside main() ?
    download_dir = args.download_destination if args.download_destination else os.getcwd()
    extract_dir = args.extract_destination if args.extract_destination else download_dir
    output_dir = args.output_destination if args.output_destination else os.path.join(download_dir, 'frames')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(download_dir, extract_dir, output_dir)
