import os
import numpy as np

def load_data(filepath):
    """Loads data from a npz file into a list of videos as numpy arrays"""
    with np.load(filepath) as f:
        frames = [arr for name, arr in f.items()]

    return list(frames[0])

def load_all_data_stacked(data_dir, skip_first=0, every_n=1):
    """Loads data from all npz files and stacks data into 4D numpy arrays"""
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    frames = [load_data(filepath) for filepath in filepaths[skip_first::every_n]]
    X = np.concatenate([np.concatenate([f[:-1] for f in frame]) for frame in frames])
    y = np.concatenate([np.concatenate([f[1:] for f in frame]) for frame in frames])

    return X, y

def load_all_data_sequences(data_dir, skip_first=0, every_n=1):
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    frames = [load_data(filepath) for filepath in filepaths[skip_first::every_n]]

    return frames

def get_splits(X, y, ratio=10):
    """Get training and test splits from numpy arrays"""

    assert X.shape[0] == y.shape[0]

    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    train_idxs = idxs[(idxs.size//ratio):]
    test_idxs = idxs[:(idxs.size//ratio)]

    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_test = X[test_idxs]
    y_test = y[test_idxs]

    return X_train, y_train, X_test, y_test

def pad_frames(frames, final_length=20):
    """Pad videos by prepending empty frames to have same length"""

    pad = np.zeros((3, 60, 80), dtype=np.int8)[np.newaxis, :, :, :]
    if frames.shape[0] <= 20:
        padstack = np.repeat(pad, final_length - frames.shape[0], axis=0)
        return np.concatenate((padstack, frames))

    else:
        return frames[:20]
