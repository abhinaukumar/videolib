import numpy as np

#: Dictionary of whitepoints of CIE Illuminants A, E, D50, and D60
illuminants = {'a': np.array([109.847, 100.0, 35.582]),  # CIE A illuminant
               'e': np.array([100.0, 100.0, 100.0]),  # CIE E (equal energy) illuminant
               'd50': np.array([96.421, 100.0, 82.519]),  # CIE D50 illuminant
               'd65': np.array([95.0456, 100.0, 108.9057])}  # CIE D65 illuminant

#: CIECAM environment parameters
envs = {'average': (1.0, 0.69, 1.0),
        'dim': (0.9, 0.59, 0.9),
        'dark': (0.8, 0.525, 0.8)}
#: CIECAM light intensities
lightintensities = {'default': 80.0, '': 318.31, 'low': 31.83, 'sdr': 100.0, 'hdr': 1000.0}  # Luminance in nits (aka cd / m^2)
#: CIECAM background intensities
bgintensities = {'default': 20, 'high': 20.0, 'low': 10.0}  # TODO: Infer default from light intensity
