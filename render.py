import os
import numpy as np
import moviepy.editor as mpe

def inference(split_type):
    """
    Loads a pretrained model and performs inference on the provided dataset
    """
    raise NotImplementedError

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
