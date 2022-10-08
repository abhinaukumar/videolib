from typing import Tuple, Callable
import warnings
from dataclasses import dataclass

from . import nonlinearities


@dataclass
class Standard:
    '''
    Class to define an image/video standard.
    The colorspace of the standard is define by its white point and primaries.
    The standard also defines opto-electrical and electro-optical functions (OETF/EOTF)
    '''
    name: str
    white_xy: Tuple[float, float]
    red_xy: Tuple[float, float]
    green_xy: Tuple[float, float]
    blue_xy: Tuple[float, float]
    linear_range: float
    oetf: Callable
    eotf: Callable

    @property
    def primaries(self):
        return {'white': self.white_xy, 'red': self.red_xy, 'green': self.green_xy, 'blue': self.blue_xy}

    @property
    def range(self):
        if self in low_bitdepth_standards:
            return (1 << 8) - 1
        elif self in high_bitdepth_standards:
            return (1 << 10) - 1
        else:
            warnings.warn('Using default range of 1.0')
            return 1
            # raise ValueError('Cannot find range for this standard')


sRGB = Standard(
    name='sRGB',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.64, 0.33), green_xy=(0.30, 0.60), blue_xy=(0.15, 0.06),
    oetf=nonlinearities.oetf_dict['sRGB'], eotf=nonlinearities.eotf_dict['sRGB'], linear_range=100
)

rec_709 = Standard(
    name='Rec.709',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.64, 0.33), green_xy=(0.30, 0.60), blue_xy=(0.15, 0.06),
    oetf=nonlinearities.oetf_dict['rec.709'], eotf=nonlinearities.eotf_dict['rec.709'], linear_range=100
)

rec_2020 = Standard(
    name='Rec.2020',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.708, 0.292), green_xy=(0.170, 0.797), blue_xy=(0.131, 0.046),
    oetf=nonlinearities.oetf_dict['rec.2020'], eotf=nonlinearities.eotf_dict['rec.2020'], linear_range=100
)

rec_2100_pq = Standard(
    name='Rec.2100.PQ',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.708, 0.292), green_xy=(0.170, 0.797), blue_xy=(0.131, 0.046),
    oetf=nonlinearities.oetf_dict['pq'], eotf=nonlinearities.eotf_dict['pq'], linear_range=10000
)

rec_2100_hlg = Standard(
    name='Rec.2100.HLG',
    white_xy=(0.3127, 0.3290),
    red_xy=(0.708, 0.292), green_xy=(0.170, 0.797), blue_xy=(0.131, 0.046),
    oetf=nonlinearities.oetf_dict['hlg'], eotf=nonlinearities.eotf_dict['hlg'], linear_range=1000
)

radiance_hdr = Standard(
    name='RadianceHDR',
    white_xy=(0.3333, 0.3333),
    red_xy=(0.640, 0.330), green_xy=(0.290, 0.600), blue_xy=(0.150, 0.060),
    oetf=nonlinearities.oetf_dict['sRGB'], eotf=nonlinearities.eotf_dict['sRGB'], linear_range=None  # Peak luminance not defined
)

supported_standards = [
    sRGB,
    rec_709,
    rec_2020,
    rec_2100_pq,
    rec_2100_hlg
]

low_bitdepth_standards = [
    sRGB,
    rec_709
]

high_bitdepth_standards = [
    rec_2020,
    rec_2100_pq,
    rec_2100_hlg
]
