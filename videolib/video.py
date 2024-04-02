import os
import subprocess
import datetime

from typing import Any, BinaryIO, Dict, Tuple, Optional, Union
from warnings import warn
import json

import numpy as np
import skvideo.io
import imageio

from . import cvt_color
from . import standards

_datatypes = ['rgb', 'linear_rgb', 'bgr', 'linear_bgr', 'yuv', 'linear_yuv', 'xyz']
TEMP_DIR = '/tmp'


class Frame:
    '''
    Class defining a frame, either of a video or an image.
    Supported native color representations: :code:`rgb`, :code:`linear_rgb`, :code:`bgr`, :code:`linear_bgr`, :code:`yuv`, :code:`linear_yuv`, :code:`xyz`.
    Access as :code:`frame.<color_space>`. For all others, use the :obj:`~videolib.cvt_color` submodule.
    '''
    def __init__(
        self,
        standard: standards.Standard,
        quantization: Optional[int] = None,
        dither: Optional[bool] = False
    ) -> None:
        '''
        Initializer.

        Args:
            standard: Coor standard to which the data conforms.
            quantization: Value to which data will be quantized and scaled back to original range.
            dither: Flag denoting whether dithering must be applied after quantization.
        '''
        self.standard: standards.Standard = standard
        self.hasdata: bool = False
        self.quantization: int = quantization
        self.dither: bool = dither
        self._base_datatype = 'yuv'  # Use non-linear YUV as the base datatype

        if self.quantization is None and self.dither is True:
            warn('Dithering is not applied when quantization is not applied.', RuntimeWarning)
        elif self.quantization is not None and (not isinstance(self.quantization, int) or self.quantization < 1):
            raise ValueError('Quantization value must be a positive integer.')
        elif self.quantization is not None and self.quantization > self.standard.range:
            raise ValueError('Quantization value must not exceed the range of the standard')

        self._quantization_step: float = (self.standard.range + 1) / self.quantization if self.quantization is not None else None

        for datatype in _datatypes:
            self.__dict__['_' + datatype] = None

    @property
    def primaries(self) -> Dict[str, Tuple[float, float]]:
        '''
        Color primaries of the Frame's standard
        '''
        return self.standard.primaries

    @property
    def width(self) -> int:
        '''
        Width of the Frame
        '''
        if self.hasdata:
            return self.yuv.shape[1]
        raise AttributeError('Width is not defined when frame has no data')

    @property
    def height(self) -> int:
        '''
        Height of the Frame
        '''
        if self.hasdata:
            return self.yuv.shape[0]
        raise AttributeError('Height is not defined when frame has no data')

    @staticmethod
    def _assert_or_make_1channel(img: np.ndarray) -> np.ndarray:
        '''
        Checks if img is a valid 1-channel image of shape (R, C).
        If not, attemps to "squeeze" the image to this shape.

        Args:
            img: 1-channel image, possibly with extra dimensions.

        Returns:
            np.ndarray: 1-channel image with no extra dimensions.

        Raises:
            ValueError: If img cannot be squeezed to 2 dimensions.
        '''
        success: bool = True

        if img.ndim < 2:
            success = False
        elif img.ndim > 2:
            if img.squeeze().ndim == 2:
                warn('Squeezing input to 2 dims.', RuntimeWarning)
                img = img.squeeze()
            else:
                success = False

        if not success:
            raise ValueError('Input cannot be interpreted as a 1-channel image.')

        return img

    @staticmethod
    def _assert_or_make_3channel(img: np.ndarray) -> np.ndarray:
        '''
        Checks if img is a valid 3-channel image of shape (R, C, 3).
        If not, attemps to "squeeze" the image to this shape.
        Args:
            img: 3-channel image, possibly with extra dimensions.

        Returns:
            np.ndarray: 3-channel image with no extra dimensions.

        Raises:
            ValueError: If img cannot be squeezed to 3 dimensions and channels.
        '''
        success: bool = True

        if img.ndim < 3:
            success = False
        elif img.ndim > 3:
            if img.squeeze().ndim == 3:
                warn('Squeezing input to 3 dims.', RuntimeWarning)
                img = img.squeeze()
            else:
                success = False

        success = success and (img.shape[-1] == 3)

        if not success:
            raise ValueError('Input cannot be interpreted as a 3-channel image.')

        return img

    @staticmethod
    def _lift_to_multichannel(img: np.ndarray, channels: int = 3) -> np.ndarray:
        '''
        "Lift" 2D array into multi-channel array, where all channels are identical.

        Args:
            img: 2D image to be "lifted"
            channels: Number of channels in the output image.

        Returns:
            np.ndarray: Lifted image.
        '''
        img = Frame._assert_or_make_1channel(img)
        return np.tile(np.expand_dims(img, -1), [1, 1, channels])

    @staticmethod
    def _get_log_average(data: Union[int, float, np.ndarray]) -> float:
        '''
        Compute log average of given data.

        Args:
            data: Data for which log average is to be computed.

        Returns:
            float: Log-average
        '''
        delta = 1e-6
        return np.exp(np.mean(np.log(data + delta)))

    def __setattr__(self, name: str, value: Any) -> None:
        '''
        Extending setattr to calculate all formats of the frame data.
        Internal copies of these formats are not allowed to be set directly.
        All other attributes are handled as usual.

        Args:
            name: Name of the attribute to be set.
            value: Value of the attribute to be set.

        Raises:
            AttributeError: If internal working copies of the frame data are set directly.
        '''
        if name in _datatypes:
            if value.dtype != 'float64':
                warn('Converting data to \'float64\'')
                value = value.astype('float64')

            for datatype in _datatypes:
                self.__dict__['_' + datatype] = None

            self.__dict__['hasdata'] = True
            value = Frame._assert_or_make_3channel(value)

            # Convert to the base datatype, from which all other types will be computed, if needed.
            if name != self._base_datatype:
                convert_to_base = cvt_color.get_conversion_function(name, self._base_datatype)
                base_value = np.clip(np.floor(convert_to_base(value, self.__dict__['standard'])), 0, self.standard.range)
            else:
                base_value = np.floor(value)

            base_value = self.quantize(base_value)
            self.__dict__['_' + self._base_datatype] = base_value

            if name != self._base_datatype:
                convert_to_input = cvt_color.get_conversion_function(self._base_datatype, name)
                self.__dict__['_' + name] = convert_to_input(base_value, self.__dict__['standard'])
        elif name in ['_' + datatype for datatype in _datatypes]:
            raise AttributeError('Class Frame does not allow setting atribute {} manually'.format(name))
        else:
            self.__dict__[name] = value

    def __getattr__(self, name: str) -> Any:
        '''
        Extending getattr to access all formats of the frame data.
        No other attributes will be found.

        Args:
            name: Name of the attribute to be found.

        Returns:
            Any: Queried attribute.

        Raises:
            AttributeError: If data has not been set yet, or any other attribute is queried.
        '''
        if name in _datatypes:
            if self.__dict__['hasdata']:
                # Convert to required datatype on demand. Saves memory.
                if self.__dict__['_' + name] is None:
                    convert_from_base = cvt_color.get_conversion_function(self._base_datatype, name)
                    self.__dict__['_' + name] = convert_from_base(self.__dict__['_' + self._base_datatype], self.__dict__['standard'])
                return self.__dict__['_' + name]

        raise AttributeError

    def quantize(self, value: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        '''
        Quantize value to have :code:`quantization` number of levels, while occupying the same input range.

        Args:
            value: Value to be quantized.

        Returns:
            Union[int, np.ndarray]: Quantized value, if :code:`quantization` attribute is specified.
        '''
        if self.quantization is not None:
            quant_value = np.round(value / self._quantization_step)
            if self.dither:
                triangular_noise = (np.random.rand(*quant_value.shape) + np.random.rand(*quant_value.shape)) - 1
                quant_value = np.clip(quant_value + triangular_noise, 0, self.quantization)
            return quant_value * self._quantization_step
        else:
            return value

    def show(self, interactive: Optional[bool] = False, **kwargs) -> None:
        '''
        Plot image using matplotlib.

        Args:
            interactive: Run matplotlib in interactive mode.
            kwargs: Passed directly to :obj:`~matplotlib.pyplot.imshow`.
        '''
        import matplotlib.pyplot as plt
        if interactive:
            plt.ion()
        plt.imshow(self.rgb / self.standard.range, **kwargs)
        if not interactive:
            plt.show()


class Video:
    '''
    Class defining a video. Reads/writes :obj:`Frame` objects to/from a file on the disk
    '''
    def __init__(
        self,
        file_path: str,
        standard: standards.Standard,
        mode: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: Optional[str] = None,
        quantization: Optional[int] = None,
        dither: Optional[bool] = False,
        out_dict: Optional[Dict] = {}
    ) -> None:
        '''
        Args:
            file_path : Path to file on the disk.
            standard: Color standard to which the data conforms.
            mode: Read/Write mode. Must be one of 'r' or 'w'.
            width: Width of each frame of the video. Required only in read mode.
            height: Height of each frame of the video. Required only in read mode.
            format: Raw/encoded format of the video.
            quantization: Value to which data will be quantized and scaled back to original range.
            dither: Flag denoting whether dithering must be applied after quantization.
            out_dict: Dictionary of \'-vodec\' and/or \'-crf\' parameters to pass to FFmpegWriter when writing encoded videos.
        '''
        self.file_path: str = file_path

        if mode not in ['r', 'w']:
            raise ValueError('Invalid choice of mode.')
        self.mode: str = mode

        self.standard = standard
        self.width: int = width
        self.height: int = height

        self.quantization: int = quantization
        self.dither: bool = dither

        self._frame_stride = 1.5 * np.dtype(self.standard.dtype).itemsize

        if self.quantization is None and self.dither is True:
            warn('Dithering is not applied when quantization is not applied.', RuntimeWarning)
        elif self.quantization is not None and (not isinstance(self.quantization, int) or self.quantization < 1):
            raise ValueError('Quantization value must be a positive integer.')
        elif self.quantization is not None and self.quantization > self.standard.range:
            raise ValueError('Quantization value must not exceed the range of the standard')

        self._range = 'Full'

        self._allowed_formats = ['raw', 'encoded', 'sdr_image', 'hdr_image']

        if format is None:
            ext = self.file_path.split('.')[-1]
            if ext == 'yuv':
                format = 'raw'
            elif ext in ['mp4', 'mov', 'avi', 'webm']:
                format = 'encoded'
            elif ext in ['jpg', 'png', 'bmp', 'tiff']:
                format = 'sdr_image'
                if self.standard not in standards.low_bitdepth_standards:
                    raise ValueError('Extension \'{ext}\' can only be used with 8-bit standards.')
            elif ext in ['hdr', 'exr']:
                format = 'hdr_image'
                if self.standard != standards.radiance_hdr:
                    raise ValueError('Extension \'{ext}\' can only be used with RadianceHDR.')
            else:
                raise ValueError(f'Format unknown for files of type \'{ext}\'')

        if format not in self._allowed_formats:
            raise ValueError(f'Invalid format. Must be one of {self._allowed_formats}.')

        self.format = format

        if (self.mode == 'r' or self.format == 'raw') and len(out_dict) != 0:
            warn('out_dict is only used when mode is \'w\' and format is \'encoded\'. Ignoring.')

        self.out_dict = {}
        self.out_dict['-vcodec'] = out_dict.pop('-vcodec', 'libx264')
        self.out_dict['-crf'] = out_dict.pop('-crf', '0')

        if len(out_dict) != 0:
            warn('Only \'-vcodec\' and \'-crf\' options are supported in out_dict. Ignoring others.')

        if self.format == 'raw':
            self._file_object: BinaryIO = open(file_path, '{}b'.format(self.mode))
        elif self.format == 'encoded':
            if self.mode == 'r':
                self._decode_encoded_video()
                self._file_object: BinaryIO = open(self._temp_path, 'rb')
            elif self.mode == 'w':
                self._file_object = skvideo.io.FFmpegWriter(file_path, outputdict=self.out_dict)
        elif 'image' in self.format:
            self._img = Frame(self.standard, self.quantization, self.dither)
            rgb = imageio.imread(file_path).astype('float64')
            if self.format == 'hdr_image':
                rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
                self._img.linear_rgb = rgb
            else:
                self._img.rgb = rgb

        self.num_frames: int = 0
        if self.mode == 'r':
            self._frames_loaded_from_next: int = 0
            self._file_object.seek(0, os.SEEK_END)
            size_in_bytes = self._file_object.tell()
            self.num_frames = size_in_bytes // int(self.width * self.height * self._frame_stride)
            self._file_object.seek(0)
            if self.width is None or self.height is None:
                raise ValueError('Must set values of width and height when reading a raw video.')
            elif 'image' in self.format:
                self.num_frames = 1
                self.width = self._img.width
                self.height = self._img.height

    @property
    def bit_depth(self) -> int:
        return self.standard.bitdepth

    @property
    def _offset(self) -> float:
        offset_dict = {
            8: 16,
            10: 64,
        }
        return offset_dict.get(self.bit_depth, 0) if self._range == 'Limited' else 0

    @property
    def _scale(self) -> float:
        scale_dict = {
            8: 255 / (235 - 16),
            10: 1023 / (940 - 64),
        }
        return scale_dict.get(self.bit_depth, 1) if self._range == 'Limited' else 1

    def _decode_encoded_video(self):
        self._temp_path = os.path.join(TEMP_DIR, f'EncodedReader_temp_{self.file_path.replace("/", "_")}_' + '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()) + '.yuv')
        json_string = subprocess.check_output(['mediainfo', '--Output=JSON', self.file_path], stdin=subprocess.DEVNULL)
        d = json.loads(json_string)
        v_track = None
        for track in d['media']['track']:
            if track['@type'] == 'Video':
                v_track = track
                break
        if v_track is None:
            raise ValueError(f'File {self.file_path} does not have a video track or MediaInfo returned unexpected output.')

        width = int(v_track['Width'])
        height = int(v_track['Height'])
        rotation = float(v_track['Rotation'])
        # Flip width and height if portrait mode
        if rotation not in [0, 180, -180]:
            width, height = height, width

        if (self.height is not None and height != self.height) or (self.width is not None and width != self.width):
            raise ValueError('Input width and height does not match video\'s dimensions.')
        else:
            self.height = height
            self.width = width

        bit_depth = int(v_track['BitDepth'])
        if self.bit_depth != bit_depth:
            raise ValueError('Video bit depth does not match standard\'s bitdepth')

        self.bytes_per_pixel = 1.5 * np.ceil(self.bit_depth / 8)
        pix_fmt = 'yuv420p'
        if self.bit_depth != 8:
            pix_fmt += f'{self.bit_depth}le'

        self._range = v_track.get('colour_range', self._range)

        cmd = [
            'ffmpeg',
            '-i', self.file_path,
            '-c:v', 'rawvideo',
            '-pix_fmt', pix_fmt,
            '-y',
            self._temp_path
        ]

        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def reset(self) -> None:
        '''
        Reset index of next to the start of the video.
        '''
        if self.mode != 'r':
            raise ValueError('Reset only defined for Video in \'r\' mode.')
        self._frames_loaded_from_next = 0

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        '''
        Returns next frame in the video.

        Returns:
            Frame: Frame object containing the frame in YUV format.

        Raises:
            StopIteration: If no more frames are to be read.
        '''
        if self.mode == 'w' or self._frames_loaded_from_next == self.num_frames:
            raise StopIteration
        frame = self.get_frame(self._frames_loaded_from_next)
        self._frames_loaded_from_next += 1
        return frame

    def get_frame(self, frame_ind: int) -> Frame:
        '''
        Read a particular frame from the video (0-based indexing).

        Args:
            frame_ind: Index of frame to be read (0-based).

        Returns:
            Frame: Frame object containing the frame in YUV format.
        '''
        if self.mode == 'w':
            raise OSError('Cannot get frame in write mode.')

        frame = Frame(self.standard, self.quantization, self.dither)
        if self.format in ['raw', 'encoded']:
            self._file_object.seek(int(self.width * self.height * frame_ind * self._frame_stride))
            y1 = np.fromfile(self._file_object, self.standard.dtype, (self.width * self.height))
            u1 = np.fromfile(self._file_object, self.standard.dtype, (self.width * self.height) >> 2)
            v1 = np.fromfile(self._file_object, self.standard.dtype, (self.width * self.height) >> 2)
            y = np.reshape(y1, (self.height, self.width)).astype('float64')
            u = np.reshape(u1, (self.height >> 1, self.width >> 1)).repeat(2, axis=0).repeat(2, axis=1).astype('float64')
            v = np.reshape(v1, (self.height >> 1, self.width >> 1)).repeat(2, axis=0).repeat(2, axis=1).astype('float64')
            yuv = np.stack((y, u, v), axis=-1)
            # Normalize the pixel values based on the determined range
            frame.yuv = (yuv - self._offset) * self._scale
        else:
            frame = self._img
        return frame

    def __getitem__(self, frame_ind: int) -> Frame:
        '''
        Read a particular frame from the video (0-based indexing).

        Args:
            frame_ind: Index of frame to be read (0-based).

        Returns:
            Frame: Frame object containing the frame in YUV format.
        '''
        if self.mode == 'w':
            raise IndexError('Cannot index video in write mode.')
        if self.format not in ['raw', 'encoded']:
            raise IndexError(f'Cannot index {self.format} format.')
        if frame_ind >= self.num_frames or frame_ind < -self.num_frames:
            raise IndexError('Frame index out of range.')

        frame_ind = frame_ind % self.num_frames
        return self.get_frame(frame_ind)

    def __setitem__(self, frame_ind: int, frame: Frame) -> None:
        '''
        Write a particular frame to the video (0-based indexing).

        Args:
            frame_ind: Index of frame to be written (0-based).
            frame: Frame to be written.
        '''
        if self.mode == 'r':
            raise OSError('Cannot assign in read mode.')
        if self.format == 'encoded':
            raise OSError('Cannot assign to encoded video.')
        frame_ind = frame_ind % self.num_frames
        if frame_ind >= self.num_frames or frame_ind < -self.num_frames:
            raise IndexError('Frame index out of range.')
        self._file_object.seek(int(self.width * self.height * frame_ind * self._frame_stride))
        self.write_frame(frame)

    def write_yuv_frame(self, yuv: np.ndarray) -> None:
        '''
        Adds YUV frame array to file on disk.

        Args:
            yuv: YUV data to be written.
        '''
        if self.mode == 'r':
            raise OSError('Cannot write YUV frame in read mode.')
        if self.format == 'raw':
            u_sub = (yuv[::2, ::2, 1] + yuv[1::2, ::2, 1] + yuv[1::2, 1::2, 1] + yuv[::2, 1::2, 1]) / 4
            v_sub = (yuv[::2, ::2, 2] + yuv[1::2, ::2, 2] + yuv[1::2, 1::2, 2] + yuv[::2, 1::2, 2]) / 4
            yuv420 = np.concatenate([np.ravel(yuv[..., 0]), np.ravel(u_sub), np.ravel(v_sub)]).astype(self.standard.dtype)
            yuv420.tofile(self._file_object)
        elif self.format == 'encoded':
            self.write_rgb_frame(cvt_color.yuv2rgb(yuv))
        self.num_frames += 1

    def write_rgb_frame(self, rgb: np.ndarray) -> None:
        '''
        Adds RGB frame array to file on disk.
        Args:
            rgb: RGB data to be written.
        '''
        if self.mode == 'r':
            raise OSError('Cannot write RGB frame in read mode.')
        if self.format == 'raw':
            yuv = cvt_color.rgb2yuv(rgb, self.standard)
            self.write_yuv_frame(yuv)
        else:
            self._file_object.writeFrame(rgb.astype(self.standard.dtype))
        self.num_frames += 1

    def write_frame(self, frame: Frame) -> None:
        '''
        Adds frame to file on disk.

        Args:
            frame: Frame object containing data to be written.
        '''
        if self.mode == 'r':
            raise OSError('Cannot write frame in read mode.')
        if self.format == 'raw':
            self.write_yuv_frame(frame.yuv)
        elif self.format == 'encoded':
            self.write_rgb_frame(frame.rgb)

    def append(self, frame: Frame) -> None:
        '''
        Appends frame to file on disk.

        Args:
            frame: Frame object containing data to be appended.
        '''
        if self.mode == 'r':
            raise OSError('Cannot append in read mode.')
        if self.format == 'raw':
            self._file_object.seek(0, 2)  # Seek end of file.
        self.write_frame(frame)

    def close(self) -> None:
        '''
        Close the file object associated with the video.
        '''
        self._file_object.close()
        if self.format == 'encoded' and self.mode == 'r':
            os.remove(self._temp_path)

    # Default behavior when entering 'with' statement
    def __enter__(self):
        return self

    # Close video file when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback):
        if self.format in ['raw', 'encoded']:
            self.close()

