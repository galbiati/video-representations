import os, sys
import numpy as np
import requests as req
import tensorflow as tf
import patoolib
import av
import tqdm

L = tf.layers

frame_to_array = lambda frame: np.array(frame.to_image())

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
        total=final_size, unit='KB', unit_scale=True,
        desc='Video archive'
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

    # clean up archive
    os.remove(filename) # set an option to keep rar archive

    return None

def load_frames(filepath, downsampler, downsample_frames=15, downsample_dims=4):
    """Load the frames from the specified video file into a numpy array"""

    video = av.open(filepath)
    raw_frames = np.array([frame_to_array(frame) for frame in video.decode(video=0)])

    if raw_frames.shape[1] != 240:
        return None

    frames = downsampler(raw_frames)

    return frames[::downsample_frames, :, :, :]

def videos_to_frames(extract_dir, output_dir, downsample_frames=15, downsample_dims=4):
    """Save downsampled video frames for all files in directory"""

    # create iterator with progress bar for action categories
    ucf_dir = os.path.join(extract_dir, 'UCF-101')
    categories = os.listdir(ucf_dir)
    category_iterator = tqdm.tqdm(categories, unit='file', desc='Converting videos to frames')

    # initialize tensorflow sesh
    input_var = tf.placeholder(dtype=tf.float32, shape=(None, 240, 320, 3), name='frames')
    pool = L.average_pooling2d(input_var, pool_size=downsample_dims, strides=downsample_dims)
    downsampler = lambda images: pool.eval({input_var: images})

    # give load_frames downsample params
    lf = lambda filepath: load_frames(filepath, downsampler=downsampler, downsample_frames=downsample_frames, downsample_dims=downsample_dims)

    # run it
    with tf.Session() as sesh:
        init = tf.global_variables_initializer()

        for category in category_iterator:

            # ignore dud files that OS X creates
            if category == '.DS_Store':
                continue

            # name and path for output file
            filename = '{}.npz'.format(category)
            filepath = os.path.join(output_dir, filename)

            # skip if file already exists
            if os.path.exists(filepath):
                continue

            # get filepaths for every video file in action category folder
            loadpath = os.path.join(ucf_dir, category)
            filenames = os.listdir(loadpath)
            filepaths = [os.path.join(loadpath, filename) for filename in filenames]

            videos = [lf(filepath) for filepath in filepaths]
            videos = [v for v in videos if v is not None]

            # name file and compress to npz

            np.savez_compressed(filepath, *videos)

            # cleanup video files
            for filepath in filepaths:
                os.remove(filepath)

    return None

def main(download_dir, extract_dir, output_dir,
            downsample_frames=15, downsample_dims=4):

    print('Downloading archive...\n')
    download_videos(download_dir)
    print('\nExtracting archive...\n')
    extract_videos(download_dir, extract_dir)

    print('\nConverting videos to numpy and reducing size...')
    videos_to_frames(
        extract_dir, output_dir,
        downsample_frames=downsample_frames, downsample_dims=downsample_dims
    )
    print('\nAll done!')
    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download the UCF-101 video data, extract it, and convert videos to downsampled arrays in npz files')

    parser.add_argument('--download_destination', type=str, help="Where the rar archive goes")
    parser.add_argument('--extract_destination', type=str, help="Where the rar contents are extracted")
    parser.add_argument('--output_destination', type=str, help="Where the npz archives are stored")
    parser.add_argument('--frames_downsample', type=str, help="Only keey every n frames")
    parser.add_argument('--dims_downsample', type=str, help="Reduce image resolution by a factor of n")
    args = parser.parse_args()

    # do the below inside main() ?
    download_dir = args.download_destination if args.download_destination else os.getcwd()
    extract_dir = args.extract_destination if args.extract_destination else download_dir
    output_dir = args.output_destination if args.output_destination else os.path.join(download_dir, 'frames')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(
        download_dir, extract_dir, output_dir,
        downsample_frames=int(args.frames_downsample),
        downsample_dims=int(args.dims_downsample)
    )
