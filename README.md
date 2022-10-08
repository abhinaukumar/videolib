# VideoLIB
_Simple video handling in Python._

# Description
The VideoLIB library provides an easy API for common image and video processing tasks like
1. Reading and writing videos from raw (YUV420p) or encoded (say, MP4) files using the `Video` class.
2. Color standard definitions (e.g. sRGB, BT.2020, etc.) for standard-accurate processing using the `Standard` class, including color gamut and transfer function definitions.
3. Color space conversion on-demand to commonly used color spaces such as RGB, YUV, CIELAB, and HSV, among others, using the `Frame` class and `cvt_color` module.
4. Color Adaptation Transforms and Uniform Color Spaces for advanced perceptually-uniform color-space conversions.
5. Convenient `CircularBuffer` class to provide an intuitive API for online temporal filtering of video frames.

# Usage
The usage examples provided below offer a glimpse into the degree to which VideoLIB simplifies video processing tasks.

## Iterating over a video framewise
```
from videolib import Video, standards

with Video('sample_video_path.mp4', standard=standards.sRGB, mode='r') as video:
    # Print video dimensions
    print(video.width, video.height, video.num_frames)

    # Show first 5 frames of the video
    max_frames = 5
    for i, frame in enumerate(video):
        if i == max_frames:
            break
        # Show RGB frame
        frame.show()

        # Show Y (grayscale) plane of YUV representation
        plt.figure()
        plt.imshow(frame.yuv[..., 0], cmap='gray')
        plt.show()
```

## Simple conversion between encoded and raw formats

```
# MP4 -> YUV (encoded -> raw)
with Video('sample_video_path.mp4', standard=standards.sRGB, mode='r') as in_video:
    with Video('sample_video_path.yuv', standard=standards.sRGB, mode='w') as out_video:
        for frame in in_video:
            out_video.append(frame)

# YUV -> MP4 (raw -> encoded)
with Video('sample_video_path.yuv', standard=standards.sRGB, mode='r') as in_video:
    with Video('sample_video_path.mp4', standard=standards.sRGB, mode='w') as out_video:
        for frame in in_video:
            out_video.append(frame)
```

## Comparing HDR videos using HDR-UCS aka Jzazbz 
```
from videolib.cvt_color import bgr2hdrucs

# Seamlessly compare HDR videos that use different encoding functions.
with Video('video1.yuv', standards.rec_2100_pq, 'r') as v1:
    with Video('video2.yuv', standards.rec_2100_hlg, 'r') as v2:
        mean_diff = 0
        for frame1, frame2 in zip(v1, v2):
            jab1 = bgr2hdrucs(frame1.bgr, frame1.standard)
            jab2 = bgr2hdrucs(frame2.bgr, frame2.standard)
            mean_diff += np.mean((jab1 - jab2)**2)
        mean_diff /= min(v1.num_frames, v2.num_frames)
        print(mean_diff)
```

# Installation
To use VideoLIB, you will need Python >= 3.7.0. In addition, using virtual environments such as `virtualenv` or `conda` is recommended. The code has been tested on Linux and it is expected to be compatible with Unix/MacOS platforms.

## Creating a virtual environment
```
python3 -m virtualenv .venv/
source .venv/bin/activate
```
## Install preqrequisites and VideoLIB
```
pip install -r requirements.txt
pip install .
```

# Issues, Suggestions, and Contributions
The goal of VideoLIB is to share with the community a tool that I build to accelerate my own video processing workflows, and one that I have found great success with. Any feedback that can improve the quality of VideoLIB for the community and myself is greatly appreciated!

Please [file an issue](https://github.com/abhinaukumar/videolib/issues) if you would like to suggest a feature, or flag any bugs/issues, and I will respond to them as promptly as I can. Contributions that add features and/or resolve any issues are also welcome! Please create a [pull request](https://github.com/abhinaukumar/videolib/pulls) with your contribution and I will review it at the earliest.

# Contact Me
If you would like to contact me personally regarding VideoLIB, please email me at either [abhinaukumar@utexas.edu](mailto:abhinaukumar@utexas.edu) or [ab.kumr98@gmail.com](mailto:ab.kumr98@gmail.com).

# License
VideoLIB is covered under the MIT License, as shown in the [LICENSE](https://github.com/abhinaukumar/videolib/blob/main/LICENSE) file.


