from typing import Tuple, Callable, Dict, Union
import warnings
from dataclasses import dataclass

import numpy as np
from . import nonlinearities


@dataclass
class Standard:
    '''
    Class to define an image/video standard.
    '''
    name: str  #: Name of the standard
    white_xy: Tuple[float, float]  #: White point in CIE xy format
    red_xy: Tuple[float, float]  #: Red primary in CIE xy format
    green_xy: Tuple[float, float]  #: Green primary in CIE xy format
    blue_xy: Tuple[float, float]  #: Blue primary in CIE xy format
    linear_range: float  #: Linear light value range
    oetf: Callable  #: Opto-Electrical (or as appropriate, inverse Electro-Optical) Transfer Function
    eotf: Callable  #: Electro-Optical Transfer Function
    bitdepth: int #: Bitdepth of the standard
    dtype: Union[str, np.dtype]  #: Data-type of pixel values being read from files

    @property
    def primaries(self) -> Dict[str, Tuple[float, float]]:
        '''
        Color primaries of the standard

        Returns:
            Dict[str, Tuple[float, float]]: Dictionary of RGB primaries and white point of the standard.
        '''
        return {'white': self.white_xy, 'red': self.red_xy, 'green': self.green_xy, 'blue': self.blue_xy}

    @property
    def range(self) -> int:
        '''
        Non-linear encoded range

        Returns:
            int: Range of non-linear encoded values. Equal to :math:`2^{\\text{bitdepth}}-1` if integer dtype, else 1.0.
        '''
        if issubclass(np.dtype(self.dtype), np.floating):
            return 1.0
        return (1 << self.bitdepth) - 1

#: The sRGB standard (Recommended default)
sRGB = Standard(
    name='sRGB',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.64, 0.33), green_xy=(0.30, 0.60), blue_xy=(0.15, 0.06),
    oetf=nonlinearities.oetf_dict['sRGB'], eotf=nonlinearities.eotf_dict['sRGB'], linear_range=100,
    bitdepth=8, dtype='uint8'
)

#: 10-bit version of the sRGB standard
sRGB_10 = Standard(
    name='sRGB.10',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.64, 0.33), green_xy=(0.30, 0.60), blue_xy=(0.15, 0.06),
    oetf=nonlinearities.oetf_dict['sRGB'], eotf=nonlinearities.eotf_dict['sRGB'], linear_range=100,
    bitdepth=10, dtype='uint16'
)

#: The ITU Rec.709 standard
rec_709 = Standard(
    name='Rec.709',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.64, 0.33), green_xy=(0.30, 0.60), blue_xy=(0.15, 0.06),
    oetf=nonlinearities.oetf_dict['rec.709'], eotf=nonlinearities.eotf_dict['rec.709'], linear_range=100,
    bitdepth=8, dtype='uint8'
)

#: 10-bit version of the ITU Rec.709 standard
rec_709_10 = Standard(
    name='Rec.709.10',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.64, 0.33), green_xy=(0.30, 0.60), blue_xy=(0.15, 0.06),
    oetf=nonlinearities.oetf_dict['rec.709'], eotf=nonlinearities.eotf_dict['rec.709'], linear_range=100,
    bitdepth=10, dtype='uint16'
)

#: The ITU Rec.2020 UHD standard
rec_2020 = Standard(
    name='Rec.2020',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.708, 0.292), green_xy=(0.170, 0.797), blue_xy=(0.131, 0.046),
    oetf=nonlinearities.oetf_dict['rec.2020'], eotf=nonlinearities.eotf_dict['rec.2020'], linear_range=100,
    bitdepth=10, dtype='uint16'
)

#: The ITU Rec.2100 standard using Perceptual Quantizer encoding
rec_2100_pq = Standard(
    name='Rec.2100.PQ',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.708, 0.292), green_xy=(0.170, 0.797), blue_xy=(0.131, 0.046),
    oetf=nonlinearities.oetf_dict['pq'], eotf=nonlinearities.eotf_dict['pq'], linear_range=10000,
    bitdepth=10, dtype='uint16'
)

#: The ITU Rec.2100 standard using Hybrid Log-Gamma encoding
rec_2100_hlg = Standard(
    name='Rec.2100.HLG',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.708, 0.292), green_xy=(0.170, 0.797), blue_xy=(0.131, 0.046),
    oetf=nonlinearities.oetf_dict['hlg'], eotf=nonlinearities.eotf_dict['hlg'], linear_range=1000,
    bitdepth=10, dtype='uint16'
)

#: The Radiance HDR format
radiance_hdr = Standard(
    name='RadianceHDR',
    white_xy=(0.3333, 0.3333),
    red_xy=(0.640, 0.330), green_xy=(0.290, 0.600), blue_xy=(0.150, 0.060),
    oetf=nonlinearities.oetf_dict['sRGB'], eotf=nonlinearities.eotf_dict['sRGB'], linear_range=None,  # Peak luminance not defined
    bitdepth=64, dtype='float64'
)

#: List of supported standards
supported_standards = [
    sRGB,
    sRGB_10,
    rec_709,
    rec_709_10,
    rec_2020,
    rec_2100_pq,
    rec_2100_hlg,
    radiance_hdr
]

#: List of 8-bit standards
standards_8bit = [
    sRGB,
    rec_709
]

#: List of 10-bit standards
standards_10bit = [
    sRGB_10,
    rec_709_10,
    rec_2020,
    rec_2100_pq,
    rec_2100_hlg
]
