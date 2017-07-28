import os
import numpy as np
import moviepy.editor as mpe

def render_movie(frame_array, output_file, fps, max_pixel=256):
    n_frames = frame_array.shape[0]
    clipped_frames = np.minimum(np.maximum(frame_array, 0), max_pixel)
    clip = mpe.ImageSequenceClip(list(clipped_frames), fps=fps)
    clip.write_videofile(output_file)
    return None
