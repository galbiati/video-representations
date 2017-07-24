import os, sys
import numpy as np
import requests as req
import rarfile as rf
import av

frame_to_array = lambda frame: np.array(frame.to_image())

def download_videos(savepath):
    """Download UCF-101 videos"""

    url = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'
    savefile = os.path.join(savepath, 'UCF101.rar')

    with req.Session() as sesh:
        g = sesh.get(url, stream=True)

    with open(savepath, 'wb') as f:
        f.write(g.content)

    return None

def extract_videos(savepath):
    """Extract UCF-101 videos from rar archive"""

    savefile = os.path.join(savepath, 'UCF101.rar')
    rarfile.RarFile.extract(savefile)

    return None

def load_frames(filepath, downsample_frames=15, downsample_image=4):
    """Load the frames from the specified video file into a numpy array"""

    video = av.open(filepath)
    frames = np.array([frame_to_array(frame) for frame in video.decode(video=0)])

    return frames[::downsample_frames, :, :, :]

def process_dir(loadpath, savepath):
    """Get frames for all files in directory"""

    # get name of category from path
    category = os.path.split(loadpath)[-1]

    # extract frames for all files
    file_to_frames = lambda filename: load_frames(os.path.join(loadpath, filename))
    all_frames = [file_to_frames(filename) for filename in os.listdir(loadpath)]

    # save frames to compressed numpy format
    output_filename = '{}.npz'.format(category)
    output_filename = os.path.join(savepath, output_filename)
    np.savez_compressed(output_filename, *all_frames)

    return None

if __name__ == '__main__':
    main()
