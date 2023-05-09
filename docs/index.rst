..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:

   Home page <self>
   API reference <_autosummary/videolib>

Welcome to VideoLIB!
====================

VideoLIB is a library for simple video handling in Python. VideoLIB library provides an easy API for common image and video processing tasks like

#. Reading and writing videos from raw (YUV420p) or encoded (say, MP4) files using the :obj:`~videolib.video.Video` class.
#. Color standard definitions (e.g. sRGB, ITU Rec.2020, etc.) for standard-accurate processing using the :obj:`~videolib.standards.Standard` class, including color gamut and transfer function definitions.
#. Color space conversion on-demand to commonly used color spaces such as RGB, YUV, CIELAB, and HSV, among others, using the :obj:`~videolib.video.Frame` class and :obj:`~videolib.cvt_color` module.
#. Color Adaptation Transforms and Uniform Color Spaces for advanced perceptually-uniform color-space conversions.
#. Convenient :obj:`~videolib.buffer.CircularBuffer` class to provide an intuitive API for online temporal filtering of video frames.