# Representation learning in videos
This repository contains models used to compress videos to frame-level latent representations.

The inspiration is vector embedding for language (characters, words, etc). These data can be represented as discrete, if large, one-hot vectors, which can be projected into a space that represents *context* information for each word. This projection is useful for building transition models for language.

Video data is in some ways fundamentally similar: videos are composed of frames, and frames likewise have context: frames are more likely to co-occur with other frames that contain the same objects, background, and so on. However, frames can't be directly projected like language data; videos are simply too high dimensional.

The aim of these models is to develop latent, frame-level embeddings for videos that can be used for frame prediction, video generation, and so on.

## Requirements
- Python >= 3.5
- ffmpeg
- rar / unrar
- requirements.txt
