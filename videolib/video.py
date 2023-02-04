import os
from typing import Any, BinaryIO, Dict, Optional, Union
from warnings import warn

import numpy as np
import skvideo.io

from . import cvt_color
from . import standards

_datatypes = ['rgb', 'linear_rgb', 'bgr', 'linear_bgr', 'yuv', 'linear_yuv', 'xyz']


class Frame:
    '''
    Class defining a frame, either of a video or an image.

    Args:
        standard (standards.Standard): Color standard to which the data conforms.
        hasdata (bool): Flag denoting whether the frame has been assigned data.
        quantization (int): Value to which data will be quantized and scaled back to original range.
        dither (bool): Flag denoting whether dithering must be applied after quantization.
        rgb (np.ndarray): If hasdata is True, contains the data in non-linear RGB format.
        bgr (np.ndarray): If hasdata is True, contains the data in non-linear BGR format.
        yuv (np.ndarray): If hasdata is True, contains the data in non-linear YUV format.
        linear_rgb (np.ndarray): If hasdata is True, contains the data in linear RGB format.
        linear_bgr (np.ndarray): If hasdata is True, contains the data in linear BGR format.
        linear_yuv (np.ndarray): If hasdata is True, contains the data in linear YUV format.
        xyz (np.ndarray): If hasdata is True, contains the data in tristimulus XYZ format.
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

        if self.standard in standards.low_bitdepth_standards:
            self._frame_dtype = np.uint8
        elif self.standard in standards.high_bitdepth_standards:
            self._frame_dtype = np.uint16
        else:
            raise ValueError('Bit depth unknown for {}'.format(self.standard.name))

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
    def primaries(self):
        return self.standard.primaries

    @property
    def width(self):
        if self.hasdata:
            return self.yuv.shape[1]
        raise AttributeError('Width is not defined when frame has no data')

    @property
    def height(self):
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
        Quantize value to have `quantization` number of levels, while occupying the same input range.

        Args:
            value: Value to be quantized.

        Returns:
            Union[int, np.ndarray]: Quantized value, if `quantization` attribute is specified.
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

        Kwargs:
            Passed directly to imshow.
        '''
        import matplotlib.pyplot as plt
        if interactive:
            plt.ion()
        plt.imshow(self.rgb / self.standard.range, **kwargs)
        if not interactive:
            plt.show()


class Video:
    '''
    Class defining a video. Reads Frames from a YUV file on the disk

    Args:
        file_path (str): Path to file on the disk.
        standard (standards.Standard): Color standard to which the data conforms.
        mode (str): Read/Write mode. Must be one of 'r' or 'w'.
        width (int): Width of each frame of the video.
        height (int): Height of each frame of the video.
        num_frames (int): Number of frames in the video.
        format (str): Raw/encoded format of the video.
        quantization (int): Value to which data will be quantized and scaled back to original range.
        dither (bool): Flag denoting whether dithering must be applied after quantization.
        out_dict (Dict): Dictionary of \'-vodec\' and/or \'-crf\' parameters to pass to FFmpegWriter when writing encoded videos.
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

        if self.standard in standards.low_bitdepth_standards:
            self._frame_dtype = np.uint8
            self._frame_stride = 1.5
        elif self.standard in standards.high_bitdepth_standards:
            self._frame_dtype = np.uint16
            self._frame_stride = 3
        else:
            raise ValueError('Bit depth unknown for {}'.format(self.standard.name))

        if self.quantization is None and self.dither is True:
            warn('Dithering is not applied when quantization is not applied.', RuntimeWarning)
        elif self.quantization is not None and (not isinstance(self.quantization, int) or self.quantization < 1):
            raise ValueError('Quantization value must be a positive integer.')
        elif self.quantization is not None and self.quantization > self.standard.range:
            raise ValueError('Quantization value must not exceed the range of the standard')

        if format is None:
            ext = self.file_path.split('.')[-1]
            if ext == 'yuv':
                format = 'raw'
            elif ext in ['mp4', 'mov', 'avi']:
                format = 'encoded'
            else:
                raise ValueError('Format unknown for files of type \'{}\''.format(ext))

        if format not in ['encoded', 'raw']:
            raise ValueError('Invalid format. Must be one of \'encoded\' or \'raw\'.')
        if self.standard in standards.high_bitdepth_standards and format != 'raw':
            raise ValueError(f'Format \'{format}\' is not supported for videos of standard {self.standard.name}.')
        else:
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
                self._file_object = skvideo.io.FFmpegReader(file_path)
            elif self.mode == 'w':
                self._file_object = skvideo.io.FFmpegWriter(file_path, outputdict=self.out_dict)

        self.num_frames: int = 0
        if self.mode == 'r':
            self._frames_loaded_from_next: int = 0
            if self.format == 'raw':
                self._file_object.seek(0, os.SEEK_END)
                size_in_bytes = self._file_object.tell()
                self.num_frames = size_in_bytes // int(self.width * self.height * self._frame_stride)
                self._file_object.seek(0)
                if self.width is None or self.height is None:
                    raise ValueError('Must set values of width and height when reading a raw video.')
            elif self.format == 'encoded':
                (self.num_frames, height, width, _) = self._file_object.getShape()  # N x H x W x C
                if (self.height is not None and height != self.height) or (self.width is not None and width != self.width):
                    raise ValueError('Input width and height does not match video\'s dimensions.')
                else:
                    self.height = height
                    self.width = width
                self._file_frame_generator = self._file_object.nextFrame()

    def reset(self) -> None:
        '''
        Reset index of next to the start of the video.
        '''
        if self.mode != 'r':
            raise ValueError('Reset only defined for Video in \'r\' mode.')
        self._frames_loaded_from_next = 0
        if self.format == 'encoded':
            self._file_object = skvideo.io.FFmpegReader(self.file_path)
            self._file_frame_generator = self._file_object.nextFrame()

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
        if self.format == 'encoded' and frame_ind != self._frames_loaded_from_next:
            raise ValueError('Encoded videos must be read sequentially.')

        frame = Frame(self.standard, self.quantization, self.dither)
        if self.format == 'raw':
            self._file_object.seek(int(self.width * self.height * frame_ind * self._frame_stride))

            y1 = np.fromfile(self._file_object, self._frame_dtype, (self.width * self.height))
            u1 = np.fromfile(self._file_object, self._frame_dtype, (self.width * self.height) >> 2)
            v1 = np.fromfile(self._file_object, self._frame_dtype, (self.width * self.height) >> 2)

            y = np.reshape(y1, (self.height, self.width)).astype('float64')
            u = np.reshape(u1, (self.height >> 1, self.width >> 1)).repeat(2, axis=0).repeat(2, axis=1).astype('float64')
            v = np.reshape(v1, (self.height >> 1, self.width >> 1)).repeat(2, axis=0).repeat(2, axis=1).astype('float64')

            frame.yuv = np.stack((y, u, v), axis=-1)
        elif self.format == 'encoded':
            frame.rgb = next(self._file_frame_generator).astype('float64')
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
            raise OSError('Cannot index video in write mode.')
        if self.format == 'encoded':
            raise OSError('Cannot index encoded video.')
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
            yuv420 = np.concatenate([np.ravel(yuv[..., 0]), np.ravel(u_sub), np.ravel(v_sub)]).astype(self._frame_dtype)
            yuv420.tofile(self._file_object)
        elif self.format == 'encoded':
            self.write_rgb_frame(cvt_color.yuv2rgb(yuv))
        self.num_frames += 1

    def write_rgb_frame(self, rgb: np.ndarray) -> None:
        if self.mode == 'r':
            raise OSError('Cannot write RGB frame in read mode.')
        if self.format == 'raw':
            yuv = cvt_color.rgb2yuv(rgb, self.standard)
            self.write_yuv_frame(yuv)
        else:
            self._file_object.writeFrame(rgb.astype(self._frame_dtype))
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

    # Default behavior when entering 'with' statement
    def __enter__(self):
        return self

    # Close video file when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
